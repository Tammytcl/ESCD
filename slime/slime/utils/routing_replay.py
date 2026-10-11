import os
import torch


ROUTING_REPLAY = None


def prepare_bshd_routing_replay(
    routes, total_lengths, max_seq_len, *, cp_size, cp_rank, tp_size, tp_rank,
    sequence_parallel, num_experts,
):
    """Match BSHD embedding padding, zig-zag CP, then sequence-parallel TP.

    Router logits flatten [local_sequence, batch, experts], not batch-major.
    Padding routes have no supervised loss; use valid, distinct expert IDs.
    """
    if len(routes) != len(total_lengths) or not routes:
        raise ValueError("R3 routes must match the nonempty Student microbatch")
    split_size = 2 * cp_size if cp_size > 1 else 1
    if max_seq_len % (split_size * (tp_size if sequence_parallel else 1)):
        raise ValueError("R3 BSHD padded length must be divisible by CP/SP splits")
    padded = []
    expected_shape = routes[0].shape[1:]
    for route, length in zip(routes, total_lengths, strict=True):
        if route.ndim != 3 or route.shape[1:] != expected_shape or route.shape[0] != length - 1:
            raise ValueError("R3 expects [Student tokens - 1, layers, topk] per sample")
        if max_seq_len < length or not 0 < route.shape[2] <= num_experts:
            raise ValueError("Invalid R3 padded length or expert count")
        pad = torch.arange(route.shape[2], dtype=route.dtype, device=route.device)
        pad = pad.view(1, 1, -1).expand(max_seq_len - route.shape[0], route.shape[1], -1)
        padded.append(torch.cat((route, pad), dim=0))
    # The model transposes BSHD inputs into SBHD before CP/SP and routing.
    result = torch.stack(padded, dim=1)
    if cp_size > 1:
        chunks = result.chunk(2 * cp_size, dim=0)
        result = torch.cat((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]), dim=0)
    if sequence_parallel:
        result = result.chunk(tp_size, dim=0)[tp_rank]
    return result.reshape(-1, *expected_shape).contiguous()


def set_routing_replay(replay):
    global ROUTING_REPLAY
    ROUTING_REPLAY = replay


class RoutingReplay:
    all_routing_replays = []

    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        RoutingReplay.all_routing_replays.append(self)

    def record(self, top_indices):
        # offload top_indices to CPU pinned memory
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def pop_forward(self):
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return top_indices.to(device=torch.cuda.current_device(), dtype=torch.long)

    def pop_backward(self):
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return top_indices.to(device=torch.cuda.current_device(), dtype=torch.long)

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self):
        self.forward_index = 0

    @staticmethod
    def clear_all():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear()

    @staticmethod
    def clear_all_forward():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear_forward()


def get_routing_replay_compute_topk(old_compute_topk):
    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            routing_replay_stage = os.environ["ROUTING_REPLAY_STAGE"]
            if routing_replay_stage == "fallthrough":
                return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            if routing_replay_stage == "record":
                probs, top_indices = old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                ROUTING_REPLAY.record(top_indices)
            elif routing_replay_stage == "replay_forward":
                top_indices = ROUTING_REPLAY.pop_forward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"[{torch.distributed.get_rank()}] top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                probs = scores.gather(1, top_indices)
            elif routing_replay_stage == "replay_backward":
                top_indices = ROUTING_REPLAY.pop_backward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                probs = scores.gather(1, top_indices)
            return probs, top_indices
        else:
            return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

    return compute_topk


def register_routing_replay(module):
    if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
        module.routing_replay = RoutingReplay()

        def pre_forward_hook(*args, **kwargs):
            set_routing_replay(module.routing_replay)

        module.register_forward_pre_hook(pre_forward_hook)
