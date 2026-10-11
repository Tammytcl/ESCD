import logging
import os
from copy import deepcopy

import wandb
import yaml

logger = logging.getLogger(__name__)


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def _get_wandb_login_key(args):
    """Return the W&B key without requiring it in Ray-serialized arguments.

    The training driver loads the local W&B config in
    ``run_with_local_wandb.sh``.  Megatron workers, however, are created as
    Ray actors with a restricted runtime environment and do not necessarily
    inherit the driver's ``WANDB_API_KEY``.  When that happens, let each
    worker read the same *path* (which is safe to pass through Ray) and load
    the credential locally.  Current launchers therefore keep the credential
    out of the Ray job command, runtime-env JSON, and ``args`` payload; a
    legacy ``--wandb-key`` caller remains supported as a fallback.
    """
    key = getattr(args, "wandb_key", None) or os.environ.get("WANDB_API_KEY")
    if key:
        return key

    config_path = os.environ.get("WANDB_LOCAL_CONFIG")
    if not config_path:
        return None

    try:
        with open(config_path, encoding="utf-8") as config_file:
            document = yaml.safe_load(config_file) or {}
        if not isinstance(document, dict):
            logger.debug("W&B config at %s is not a mapping", config_path)
            return None
        config = document.get("wandb", document)
        if not isinstance(config, dict):
            logger.debug("W&B config section at %s is not a mapping", config_path)
            return None
        key = config.get("api_key")
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        # Do not make a missing/unreadable optional secret file look like a
        # distributed-training failure.  The subsequent W&B init will report
        # the actionable authentication error if online tracking is enabled.
        logger.debug("Unable to load W&B key from %s: %s", config_path, exc)
        return None

    return str(key) if key else None


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    wandb_key = _get_wandb_login_key(args)
    if (not offline) and wandb_key is not None:
        wandb.login(key=wandb_key, host=args.wandb_host)

    # Prepare wandb init parameters
    # add random 6 length string with characters
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id


def reinit_wandb_primary_with_open_metrics(args, router_addr):
    """Re-initialize the primary W&B run with open metrics endpoints.

    The primary wandb init happens before rollout servers start (to obtain
    ``wandb_run_id`` for secondary processes).  This function is called
    *after* servers are up so the router address is available for scraping
    SGLang Prometheus metrics via the primary process's stats monitor.
    """
    if not args.use_wandb or _is_offline_mode(args):
        return
    if getattr(args, "wandb_mode", None) == "disabled":
        return
    if router_addr is None:
        return
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    import sglang_router

    if "slime" not in sglang_router.__version__:
        logger.warning(
            "Only customized sglang_router from https://github.com/zhuzilin/sgl-router supports uploading metrics."
        )
        return

    logger.info(f"Re-initializing primary W&B with SGLang metrics at {router_addr}.")

    wandb.finish()

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(
            mode="shared",
            x_primary=True,
            x_stats_open_metrics_endpoints={
                "sgl_engine": f"{router_addr}/engine_metrics",
            },
            x_stats_open_metrics_filters={
                "sgl_engine.*": {},
            },
        ),
    }

    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)
    _init_wandb_common()


def _compute_config_for_logging(args):
    output = _args_to_config_dict(args)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    if getattr(args, "use_critic", False):
        critic_args = _get_role_args_for_logging(args, role="critic")
        output.update(_prefix_config_keys(_args_to_config_dict(critic_args), "critic"))

    return output


def _args_to_config_dict(args):
    # ``wandb_key`` is a legacy CLI escape hatch.  Never upload it as part of
    # the run config: older launchers may still populate the field even when
    # the current secret-injection path uses ``WANDB_LOCAL_CONFIG``.
    output = deepcopy(args.__dict__)
    output.pop("wandb_key", None)
    return output


def _prefix_config_keys(config, prefix):
    return {f"{prefix}/{key}": value for key, value in config.items()}


def _get_role_args_for_logging(args, role):
    if getattr(args, "megatron_config_path", None) is None:
        return args

    from slime.utils.arguments import parse_megatron_role_args

    return parse_megatron_role_args(args, args.megatron_config_path, role=role)


def _compute_secondary_config_for_logging(args, role=None):
    config = _args_to_config_dict(args)
    if role == "critic":
        return _prefix_config_keys(config, "critic")
    return config


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, role=None):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    wandb_key = _get_wandb_login_key(args)
    if (not offline) and wandb_key is not None:
        wandb.login(key=wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": _compute_secondary_config_for_logging(args, role=role),
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    # W&B only accepts a wildcard as the final suffix of a metric pattern.
    # This prefix pattern also matches nested MoE keys such as
    # train/moe/layer_XX/cv; train/moe/*/* is rejected by W&B 0.27.0 and
    # would abort training during tracker initialization.
    wandb.define_metric("train/moe/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
