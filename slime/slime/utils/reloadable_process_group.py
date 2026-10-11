import logging
import os
from contextlib import ExitStack, contextmanager

import torch
import torch.distributed as dist
from torch_memory_saver import torch_memory_saver

from slime.utils.memory_utils import available_memory, clear_memory, print_memory

logger = logging.getLogger(__name__)

old_new_group_dict = {}

_COMM_MEMORY_CHECK_SKIP_OPS = {
    "all_gather_into_tensor",
    "allgather_into_tensor_coalesced",
    "barrier",
    "broadcast_object_list",
    "reduce_scatter_tensor",
    "reduce_scatter_tensor_coalesced",
    "all_to_all_single",
    "isend",
    "irecv",
}


class _CombinedWork:
    """Small Work-compatible handle for the non-coalesced fallback.

    Some NCCL builds expose ``reduce_scatter_tensor`` but not its coalesced
    variant.  The c10d coalescing manager expects one object with ``wait()``
    after dispatching the coalesced operation, so collect the individual
    handles behind the same minimal interface.
    """

    def __init__(self, works):
        self._works = [work for work in works if work is not None]

    def wait(self, *args, **kwargs):
        result = None
        for work in self._works:
            result = work.wait(*args, **kwargs)
        return result

    def is_completed(self):
        return all(work.is_completed() for work in self._works)

    def is_success(self):
        return all(work.is_success() for work in self._works)


_reduce_scatter_fallback_logged = False
_allgather_fallback_logged = False


def _should_check_memory_for_comm(op_name):
    return op_name not in _COMM_MEMORY_CHECK_SKIP_OPS


@contextmanager
def _disable_memory_saver_if_available():
    """Disable TorchMemorySaver only in processes configured to use it.

    Separate actor/rollout deployments do not install the memory-saver
    ``LD_PRELOAD`` hook.  Calling ``disable()`` there tries to initialize the
    library and raises before NCCL warmup starts.  Colocated/offload processes
    still enter the real context; only the library's explicit missing-preload
    assertion is treated as an inactive configuration.
    """

    with ExitStack() as stack:
        try:
            stack.enter_context(torch_memory_saver.disable())
        except AssertionError as error:
            # Depending on the offload transition, the preload library can
            # be installed while its "interesting region" is inactive.  In
            # that state torch_memory_saver.disable() raises a second,
            # version-specific assertion instead of acting as a no-op.  NCCL
            # group warmup is safe without the context in either inactive
            # case, so continue just as we do when the preload hook is absent.
            error_text = str(error)
            if (
                "invalid LD_PRELOAD" not in error_text
                and "disable() should be called only when tms is active" not in error_text
            ):
                raise
            logger.info(
                "TorchMemorySaver is inactive in this process; warming NCCL groups without disable(): %s",
                error,
            )
        yield


def monkey_patch_torch_dist():
    pid = os.getpid()
    if pid in old_new_group_dict:
        assert dist.old_new_group == old_new_group_dict[pid]
        return

    logger.info("Applying monkey patch to torch.distributed")

    old_new_group = dist.new_group
    old_new_group_dict[pid] = old_new_group
    dist.old_new_group = old_new_group

    def new_group(*args, **kwargs):
        # Ray actors use a global distributed rank while CUDA devices are
        # local to each worker.  If NCCL is left unbound, the first lazy P2P
        # (PP/CP send/recv) communicator guesses ``global_rank`` as its device
        # and can hang before the first training microbatch.  Bind every NCCL
        # subgroup to this process' actual current CUDA device at creation
        # time; this is also honored when the subgroup is recreated after an
        # offload pause.
        backend = kwargs.get("backend")
        if backend is None and len(args) >= 3:
            backend = args[2]
        if backend == "nccl" and "device_id" not in kwargs:
            try:
                kwargs["device_id"] = torch.device("cuda", torch.cuda.current_device())
            except (AssertionError, RuntimeError):
                # CPU-only/unit-test processes may still exercise the wrapper;
                # let the backend choose its normal device in that case.
                pass
        group = old_new_group(*args, **kwargs)
        # skip none nccl group.
        if len(args) >= 3 and args[2] == "gloo" or "backend" in kwargs and kwargs["backend"] == "gloo":
            return group

        # Get ranks from arguments
        if len(args) >= 1 and args[0] is not None:
            ranks = args[0]
        elif "ranks" in kwargs and kwargs["ranks"] is not None:
            ranks = kwargs["ranks"]
        else:
            # If no ranks specified, use all ranks in world
            ranks = list(range(dist.get_world_size()))

        if len(ranks) == 1:
            return group

        group_desc = kwargs.get("group_desc")
        group = ReloadableProcessGroup(group, ranks, group_desc=group_desc)
        return group

    dist.new_group = new_group

    def get_new_query_function(func):
        """Wrap query functions (get_rank, get_world_size, etc.) without memory check."""

        def new_function(*args, **kwargs):
            args = tuple([arg.group if isinstance(arg, ReloadableProcessGroup) else arg for arg in args])
            kwargs = {k: (v.group if isinstance(v, ReloadableProcessGroup) else v) for k, v in kwargs.items()}
            return func(*args, **kwargs)

        return new_function

    def get_new_comm_function(func, op_name=None):
        """Wrap communication functions with memory check."""

        def new_function(*args, **kwargs):
            args = tuple([arg.group if isinstance(arg, ReloadableProcessGroup) else arg for arg in args])
            kwargs = {k: (v.group if isinstance(v, ReloadableProcessGroup) else v) for k, v in kwargs.items()}
            check_memory = True if op_name is None else _should_check_memory_for_comm(op_name)
            with _wrap_low_level_call(check_memory=check_memory):
                return func(*args, **kwargs)

        return new_function

    dist.get_rank = get_new_query_function(dist.get_rank)
    dist.get_world_size = get_new_query_function(dist.get_world_size)
    dist.get_backend = get_new_query_function(dist.get_backend)
    dist.get_global_rank = get_new_query_function(dist.get_global_rank)
    dist.get_group_rank = get_new_query_function(dist.get_group_rank)
    dist.get_process_group_ranks = get_new_query_function(dist.get_process_group_ranks)

    dist.all_reduce = get_new_comm_function(dist.all_reduce)
    dist.all_gather = get_new_comm_function(dist.all_gather)
    dist.all_gather_into_tensor = get_new_comm_function(
        dist.all_gather_into_tensor, "all_gather_into_tensor"
    )
    dist.all_gather_object = get_new_comm_function(dist.all_gather_object)
    dist.all_to_all = get_new_comm_function(dist.all_to_all)
    dist.all_to_all_single = get_new_comm_function(dist.all_to_all_single, "all_to_all_single")
    dist.broadcast = get_new_comm_function(dist.broadcast)
    dist.broadcast_object_list = get_new_comm_function(
        dist.broadcast_object_list, "broadcast_object_list"
    )
    dist.reduce = get_new_comm_function(dist.reduce)
    dist.reduce_scatter = get_new_comm_function(dist.reduce_scatter)
    dist.reduce_scatter_tensor = get_new_comm_function(
        dist.reduce_scatter_tensor, "reduce_scatter_tensor"
    )
    dist.scatter = get_new_comm_function(dist.scatter)
    dist.gather = get_new_comm_function(dist.gather)
    dist.barrier = get_new_comm_function(dist.barrier, "barrier")
    dist.send = get_new_comm_function(dist.send)
    dist.recv = get_new_comm_function(dist.recv)
    dist._coalescing_manager = get_new_comm_function(dist._coalescing_manager)

    # p2p
    old_isend = dist.isend
    old_irecv = dist.irecv

    dist.isend = get_new_comm_function(dist.isend, "isend")
    dist.irecv = get_new_comm_function(dist.irecv, "irecv")

    def get_new_p2pop_function(func):
        def new_function(*args, **kwargs):
            def convert(arg):
                if isinstance(arg, ReloadableProcessGroup):
                    return arg.group
                elif arg == dist.isend:
                    arg = old_isend
                elif arg == dist.irecv:
                    arg = old_irecv
                return arg

            args = (convert(arg) for arg in args)
            kwargs = {k: convert(v) for k, v in kwargs.items()}
            return func(*args, **kwargs)

        return new_function

    dist.P2POp.__new__ = get_new_p2pop_function(dist.P2POp.__new__)
    dist.P2POp.__init__ = get_new_p2pop_function(dist.P2POp.__init__)


class ReloadableProcessGroup(torch.distributed.ProcessGroup):
    GROUPS = {}

    def __init__(self, group, ranks, *, group_desc=None):
        super().__init__(
            rank=dist.get_rank(group),
            size=dist.get_world_size(group),
        )
        self.group = group
        self.group_info = {
            "ranks": tuple(int(rank) for rank in ranks),
            "group_desc": str(group_desc or "unnamed"),
        }
        pid = os.getpid()
        if pid not in ReloadableProcessGroup.GROUPS:
            ReloadableProcessGroup.GROUPS[pid] = []
        ReloadableProcessGroup.GROUPS[pid].append(self)

    def __getattr__(self, name):
        return getattr(self.group, name)

    @staticmethod
    def warmup_process_groups():
        """Eagerly initialize every recorded NCCL communicator in creation order.

        Megatron creates overlapping TP/EP/ETP groups in one canonical order,
        but NCCL normally initializes each communicator lazily on first use.
        Different MoE ranks can consequently rendezvous different groups in
        the first forward.  A member-only barrier here turns that implicit,
        workload-dependent ordering into an explicit initialization phase.
        """

        pid = os.getpid()
        reloadable_groups = ReloadableProcessGroup.GROUPS.get(pid, [])
        if not reloadable_groups:
            logger.info("No reloadable NCCL process groups to warm up in pid %s", pid)
            return
        global_rank = dist.get_rank()
        logger.info(
            "Warming up %s NCCL process groups in creation order on global rank %s",
            len(reloadable_groups),
            global_rank,
        )
        # The actor creates one CPU world group before Megatron groups.  Use it
        # to stop non-members from racing ahead to a later NCCL communicator.
        from slime.utils.distributed_utils import get_gloo_group

        world_fence = get_gloo_group()
        with _disable_memory_saver_if_available():
            for group_index, reloadable_group in enumerate(reloadable_groups):
                ranks = reloadable_group.group_info["ranks"]
                if reloadable_group.group is None:
                    raise RuntimeError(
                        f"Cannot warm up destroyed NCCL process group index={group_index}"
                    )
                dist.barrier(group=world_fence)
                if global_rank in ranks:
                    group_desc = reloadable_group.group_info["group_desc"]
                    logger.info(
                        "NCCL process-group warmup START index=%s desc=%s size=%s ranks=%s",
                        group_index,
                        group_desc,
                        len(ranks),
                        ranks,
                    )
                    # The Ray rank is global across the actor placement, while
                    # CUDA device ids are local to each worker.  Passing the
                    # local device explicitly avoids NCCL's global-rank guess
                    # during the eager subgroup barrier.  The previous code
                    # could emit ProcessGroupNCCL.cpp:5188 and defer the same
                    # ambiguity to the first PP/CP send/recv.
                    dist.barrier(
                        group=reloadable_group.group,
                        device_ids=[torch.cuda.current_device()],
                    )
                    logger.info(
                        "NCCL process-group warmup COMPLETE index=%s desc=%s size=%s",
                        group_index,
                        group_desc,
                        len(ranks),
                    )
                dist.barrier(group=world_fence)
        logger.info("NCCL process-group warmup COMPLETE on global rank %s", global_rank)

    @staticmethod
    def destroy_process_groups():
        pid = os.getpid()
        for reloadable_group in ReloadableProcessGroup.GROUPS.get(pid, []):
            if reloadable_group.group is None:
                continue
            try:
                dist.destroy_process_group(reloadable_group.group)
            except ValueError as e:
                logger.warning(
                    f"Process group already invalid/destroyed; skipping cleanup. Exception: {e}",
                    exc_info=True,
                )

            del reloadable_group.group
            reloadable_group.group = None

    @staticmethod
    def reload_process_groups():
        pid = os.getpid()
        reloadable_groups = ReloadableProcessGroup.GROUPS.get(pid, [])
        logger.info(f"Reloading {len(reloadable_groups)} process groups in pid {pid}")
        old_new_group = old_new_group_dict.get(pid)
        # NCCL communicator initialization must not allocate through the
        # memory saver: saver-managed virtual memory is unmapped at the next
        # pause(), which leaves live communicators pointing at unmapped
        # buffers (multi-node NET transports crash or wedge on first use).
        # disable() routes every allocation in this block to real device
        # memory, matching the invariant update_weights already relies on.
        with _disable_memory_saver_if_available():
            for reloadable_group in reloadable_groups:
                if reloadable_group.group is not None:
                    continue
                ranks = reloadable_group.group_info["ranks"]
                group = old_new_group(
                    ranks=ranks,
                    backend="nccl",
                    group_desc=reloadable_group.group_info["group_desc"],
                )
                reloadable_group.group = group
            # Recreated communicators are lazy too.  Warm them only after all
            # ranks have recreated the complete canonical group list.
            ReloadableProcessGroup.warmup_process_groups()

    def rank(self) -> int:
        return self.group.rank()

    def size(self) -> int:
        return self.group.size()

    def name(self) -> str:
        return self.group.name()

    def shutdown(self) -> None:
        if self.group is not None:
            self.group.shutdown()

    def abort(self) -> None:
        if self.group is not None:
            self.group.abort()

    def _fwd(self, method, *args, **kwargs):
        inner = self.group
        if inner is None:
            raise RuntimeError("ReloadableProcessGroup: inner PG is None, call reload() first.")
        with _wrap_low_level_call(check_memory=_should_check_memory_for_comm(method)):
            return getattr(inner, method)(*args, **kwargs)

    def _fwd_query(self, method, *args, **kwargs):
        """Forward non-communication calls without memory check."""
        inner = self.group
        if inner is None:
            raise RuntimeError("ReloadableProcessGroup: inner PG is None, call reload() first.")
        return getattr(inner, method)(*args, **kwargs)

    def barrier(self, *a, **kw):
        return self._fwd("barrier", *a, **kw)

    def broadcast(self, *a, **kw):
        return self._fwd("broadcast", *a, **kw)

    def allreduce(self, *a, **kw):
        return self._fwd("allreduce", *a, **kw)

    def allreduce_coalesced(self, *a, **kw):
        return self._fwd("allreduce_coalesced", *a, **kw)

    def reduce(self, *a, **kw):
        return self._fwd("reduce", *a, **kw)

    def allgather(self, *a, **kw):
        return self._fwd("allgather", *a, **kw)

    def _allgather_base(self, *a, **kw):
        return self._fwd("_allgather_base", *a, **kw)

    def allgather_coalesced(self, *a, **kw):
        return self._fwd("allgather_coalesced", *a, **kw)

    def allgather_into_tensor_coalesced(self, outputs, inputs, opts=None):
        """Forward coalesced all-gather with an NCCL compatibility fallback.

        The distributed optimizer uses this newer c10d entrypoint for
        parameter synchronization.  The DLC NCCL build may expose
        ``_allgather_base`` but reject the coalesced variant, so issue one
        base all-gather per bucket when needed.
        """
        global _allgather_fallback_logged
        try:
            if opts is None:
                return self._fwd("allgather_into_tensor_coalesced", outputs, inputs)
            return self._fwd("allgather_into_tensor_coalesced", outputs, inputs, opts)
        except RuntimeError as error:
            if "does not support allgather_into_tensor_coalesced" not in str(error):
                raise
            if not _allgather_fallback_logged:
                logger.warning(
                    "NCCL lacks allgather_into_tensor_coalesced; "
                    "falling back to per-bucket _allgather_base operations."
                )
                _allgather_fallback_logged = True

            works = []
            for output, input in zip(outputs, inputs):
                if opts is None:
                    works.append(self._fwd("_allgather_base", output, input))
                else:
                    works.append(self._fwd("_allgather_base", output, input, opts))
            return _CombinedWork(works)

    def gather(self, *a, **kw):
        return self._fwd("gather", *a, **kw)

    def scatter(self, *a, **kw):
        return self._fwd("scatter", *a, **kw)

    def reduce_scatter(self, *a, **kw):
        return self._fwd("reduce_scatter", *a, **kw)

    def _reduce_scatter_base(self, *a, **kw):
        return self._fwd("_reduce_scatter_base", *a, **kw)

    def reduce_scatter_tensor_coalesced(self, outputs, inputs, opts=None):
        """Forward coalesced reduce-scatter, with a backend compatibility path.

        The torch 2.11 c10d coalescing manager calls this ProcessGroup method
        directly.  The NCCL build used by the current DLC image implements
        ``_reduce_scatter_base`` but raises when the newer coalesced entrypoint
        is used.  Fall back to one base operation per bucket in that case while
        retaining the original async/sync setting from ``opts``.
        """
        global _reduce_scatter_fallback_logged
        try:
            return self._fwd("reduce_scatter_tensor_coalesced", outputs, inputs, opts)
        except RuntimeError as error:
            if "does not support reduce_scatter_tensor_coalesced" not in str(error):
                raise
            if not _reduce_scatter_fallback_logged:
                logger.warning(
                    "NCCL lacks reduce_scatter_tensor_coalesced; "
                    "falling back to per-bucket _reduce_scatter_base operations."
                )
                _reduce_scatter_fallback_logged = True

            if opts is None:
                opts = torch.distributed.ReduceScatterOptions()
            works = [
                self._fwd("_reduce_scatter_base", output, input, opts)
                for output, input in zip(outputs, inputs)
            ]
            return _CombinedWork(works)

    def alltoall_base(self, *a, **kw):
        return self._fwd("alltoall_base", *a, **kw)

    def alltoall(self, *a, **kw):
        return self._fwd("alltoall", *a, **kw)

    def send(self, *a, **kw):
        return self._fwd("send", *a, **kw)

    def recv(self, *a, **kw):
        return self._fwd("recv", *a, **kw)

    def recv_anysource(self, *a, **kw):
        return self._fwd("recv_anysource", *a, **kw)

    def _start_coalescing(self, *a, **kw):
        return self._fwd_query("_start_coalescing", *a, **kw)

    def _end_coalescing(self, *a, **kw):
        return self._fwd("_end_coalescing", *a, **kw)

    def _get_backend_name(self):
        return self._fwd_query("_get_backend_name")

    def _get_backend(self, *a, **kw):
        return self._fwd_query("_get_backend", *a, **kw)

    def _set_default_backend(self, *a, **kw):
        return self._fwd_query("_set_default_backend", *a, **kw)

    @property
    def bound_device_id(self):
        return self.group.bound_device_id

    @bound_device_id.setter
    def bound_device_id(self, dev):
        self.group.bound_device_id = dev


def destroy_process_groups():
    """Destroy all reloadable process groups."""
    ReloadableProcessGroup.destroy_process_groups()


def reload_process_groups():
    """Reload all reloadable process groups."""
    ReloadableProcessGroup.reload_process_groups()


def warmup_process_groups():
    """Eagerly initialize all reloadable NCCL communicators."""

    ReloadableProcessGroup.warmup_process_groups()


@contextmanager
def _wrap_low_level_call(check_memory=True):
    try:
        if check_memory:
            mem_info = available_memory()
            if mem_info["free_GB"] < 3:
                clear_memory()
        yield
    except Exception as e:
        mem_info = print_memory("after torch distributed error")
        e.add_note(f"{mem_info=}")
        raise
