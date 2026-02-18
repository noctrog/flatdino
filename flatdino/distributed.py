from typing import Callable, Literal, Type, TypeVar, Iterable
from pathlib import Path
import collections
import itertools
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import flax.nnx as nnx
import grain.python as grain
import orbax.checkpoint as ocp
from absl import logging


# =============================================================================
# Distributed Training Utilities
# =============================================================================


def is_primary_host() -> bool:
    """Returns True if this is the primary host (process 0) in multi-host setup."""
    return jax.process_index() == 0


def init_distributed():
    """Initialize JAX distributed for multi-host training.

    Must be called before any JAX operations (including jnp array creation).
    For TPU pods, JAX auto-detects the distributed setup. For GPU clusters,
    you may need to set environment variables like COORDINATOR_ADDRESS.
    """
    jax.distributed.initialize()
    logging.info(
        f"Initialized distributed JAX: process {jax.process_index()} of {jax.process_count()}"
    )


# =============================================================================
# Server Utilities
# =============================================================================


def is_running_on_gcp():
    """This function checks if the current script is running on a Google Cloud computer."""
    import requests

    try:
        response = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/id",
            headers={"Metadata-Flavor": "Google"},
            timeout=1,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


# =============================================================================
# Model Utilities
# =============================================================================


T = TypeVar("T")


class Restorable:
    """A protocol for classes that can be restored from an orbax checkpoint.

    In case of sharding, this assumes that the model is fully replicated across
    all devices.
    """

    @classmethod
    def restore(
        cls: Type[T],
        mngr: ocp.CheckpointManager,
        step: int,
        name: str,
        mesh: jax.sharding.Mesh | None = None,
        *args,
        **kwargs,
    ) -> T:
        model = nnx.eval_shape(lambda: cls(*args, **kwargs, rngs=nnx.Rngs(0)))
        graphdef, state = nnx.split(model)
        pure_state = nnx.to_pure_dict(state)
        restore_args = jax.tree.map(
            lambda x: ocp.ArrayRestoreArgs(sharding=NamedSharding(mesh, P())), pure_state
        )
        pure_state = mngr.restore(
            step,
            args=ocp.args.Composite(
                **{name: ocp.args.PyTreeRestore(pure_state, restore_args=restore_args)}
            ),
        )[name]
        nnx.replace_by_pure_dict(state, pure_state)
        return nnx.merge(graphdef, state)


class NNXIdentity(nnx.Module):
    def __init__(self):
        # The dummy is needed to enable checkpointing of this module
        self.dummy = nnx.Variable(jnp.array(1))

    def __call__(self, x: jax.Array, *args, **kwargs) -> jax.Array:
        return x


class TrainingProfiler:
    """Helper to manage JAX profiler server for training.

    The profiler can operate in three modes:
    - "disabled": No profiling. The step() method is a no-op.
    - "always": Start the profiler server immediately and keep it running
      indefinitely. Useful for ad-hoc profiling during development.
    - "window": Start the profiler at start_step and stop it at stop_step.
      Useful for profiling a specific phase of training (e.g., after warmup).

    Usage:
        # Always-on profiling
        profiler = TrainingProfiler(mode="always", port=7777)

        # Window-based profiling (steps 100-500)
        profiler = TrainingProfiler(mode="window", port=7777, start_step=100, stop_step=500)

        # In training loop
        for step in range(num_steps):
            train_step(...)
            profiler.step(step)
    """

    def __init__(
        self,
        mode: Literal["disabled", "always", "window"] = "disabled",
        port: int = 7777,
        start_step: int = 10,
        stop_step: int = 200,
    ):
        """Initialize the TrainingProfiler.

        Args:
            mode: Profiling mode - "disabled", "always", or "window".
            port: Port for the profiler server.
            start_step: Step to start profiling (only used in "window" mode).
            stop_step: Step to stop profiling (only used in "window" mode).
        """
        self.mode = mode
        self.port = port
        self.start_step = start_step
        self.stop_step = stop_step
        self._started = False

        # Start immediately in "always" mode
        if self.mode == "always":
            logging.info(f"Starting JAX profiler server on port {self.port} (always-on mode)")
            jax.profiler.start_server(self.port)
            self._started = True

    def step(self, step: int):
        """Called each training step to manage profiler state.

        In "window" mode, starts the profiler when step >= start_step and
        stops it when step >= stop_step. In other modes, this is a no-op.

        Args:
            step: Current training step.
        """
        if self.mode != "window":
            return

        if not self._started and step >= self.start_step and step < self.stop_step:
            logging.info(f"Starting JAX profiler server on port {self.port} (window mode)")
            jax.profiler.start_server(self.port)
            self._started = True

        elif self._started and step >= self.stop_step:
            logging.info("Stopping JAX profiler server")
            jax.profiler.stop_server()
            self._started = False


# This is needed because when injecting the momentum in optax, it expects the number to be on the GPU
# TODO: find more elegant solution
def cosine_scheduler_jax(start: float, end: float, num_iter: int) -> Callable:
    def interpolate_jax(i: int) -> jax.Array:
        progress = i / num_iter

        cosine_value = end + (start - end) * 0.5 * (1 + jnp.cos(jnp.pi * progress))
        value = jnp.where(i < 0, start, jnp.where(i < num_iter, cosine_value, end))
        return value

    return interpolate_jax


def prefetch_to_mesh(
    iterator: Iterable,
    size: int,
    mesh: jax.sharding.Mesh,
    xs_spec=P("data"),
    trim: bool = False,
    pad_to: int | None = None,
):
    """Assumes that all objects of the PyTree are jax.Arrays and have a batch dimension.

    In multi-host setups, each host's iterator should yield only that host's shard of
    the global batch. The data is then placed onto local devices and combined into a
    globally-sharded array.
    """
    num_local_devices = len(mesh.local_devices)
    warned = False
    queue = collections.deque()

    def _prefetch(xs):
        nonlocal warned
        shape = np.shape(xs)

        # Handle scalars - replicate across all devices
        if len(shape) == 0:
            sharding = NamedSharding(mesh, P())
            return jax.device_put(xs, sharding)

        # Trim batch to be divisible by local devices (each host handles its portion)
        if trim and pad_to is None:
            trimmed_bs = (shape[0] // num_local_devices) * num_local_devices
            if not warned and trimmed_bs != shape[0]:
                logging.warning(
                    f"Batch original size ({shape[0]}) cannot be sharded. Trimming to {trimmed_bs}"
                )
                warned = True
            xs = xs[:trimmed_bs, ...]
            shape = np.shape(xs)

        # Create sharding for the global array
        sharding = NamedSharding(mesh, xs_spec)

        # In multi-host: each process provides its local data, JAX assembles globally
        if jax.process_count() > 1:
            return jax.make_array_from_process_local_data(sharding, xs)
        else:
            return jax.device_put(xs, sharding)

    def _pad_if_needed(x):
        shape = np.shape(x)
        if pad_to is None or len(shape) == 0:
            return x
        batch = shape[0]
        remainder = batch % pad_to
        if remainder == 0:
            return x
        pad = pad_to - remainder
        pad_block = np.repeat(np.asarray(x[-1:]), pad, axis=0)
        return np.concatenate([np.asarray(x), pad_block], axis=0)

    # Enqueues *up to* `n` elements from the iterator
    def enqueue(n):
        for data in itertools.islice(iterator, n):
            valid = None
            if pad_to is not None:
                leaves = jax.tree.leaves(data)
                if leaves:
                    leaf = leaves[0]
                    if hasattr(leaf, "shape") and leaf.shape:
                        valid = int(leaf.shape[0])
                data = jax.tree.map(_pad_if_needed, data)
                if isinstance(data, dict) and valid is not None:
                    data = {**data, "_valid_size": valid}
            queue.append(jax.tree.map(_prefetch, data))

    enqueue(size)
    while queue:
        yield queue.popleft()
        enqueue(1)


def block_causal_mask(block_size: int, num_blocks: int, dtype=jnp.bool_) -> jax.Array:
    frame_mask = jnp.tril(jnp.ones((num_blocks, num_blocks), dtype=jnp.int8))
    patch_mask = jnp.ones((block_size, block_size), dtype=jnp.int8)
    mask = jnp.kron(frame_mask, patch_mask).astype(jnp.bool_)
    return mask[jnp.newaxis, jnp.newaxis, :, :]


def jax_unstack(x: jax.Array, axis: int = 0):
    return [jax.lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])]


# =============================================================================
# Checkpoint Path Utilities
# =============================================================================


def determine_save_path(
    checkpoint_enabled: bool,
    checkpoint_dir: Path | None,
    default_path: Path | str,
    gcs_bucket: str | None = None,
) -> Path | str | None:
    """Determine the checkpoint save path, with optional GCS support.

    Args:
        checkpoint_enabled: Whether checkpointing is enabled.
        checkpoint_dir: User-specified checkpoint directory (overrides default).
        default_path: Default path to use if checkpoint_dir is not specified.
        gcs_bucket: GCS bucket name. When set, converts the path to gs://{bucket}/{path}.

    Returns:
        The save path as a Path (local) or string (GCS), or None if checkpointing is disabled.
    """
    if not checkpoint_enabled:
        logging.warning("Not storing checkpoints of model")
        return None

    if checkpoint_dir is not None:
        save_path: Path | str = checkpoint_dir
    else:
        save_path = Path(default_path) if isinstance(default_path, str) else default_path

    # If GCS bucket is specified, convert to GCS path
    if gcs_bucket is not None:
        relative_path = str(save_path) if isinstance(save_path, Path) else save_path
        # Remove leading slashes and gs:// prefix if accidentally included
        relative_path = relative_path.lstrip("/")
        if relative_path.startswith("gs://"):
            relative_path = relative_path.split("/", 3)[-1] if "/" in relative_path else ""
        save_path = f"gs://{gcs_bucket}/{relative_path}"
        logging.info(f"Using GCS path for checkpoints: {save_path}")

    return save_path


# =============================================================================
# Checkpoint Restore Utilities
# =============================================================================


def compute_restore_path(
    restore_arg: Path | Literal["default"],
    default_path: str,
    gcs_bucket: str | None = None,
) -> str:
    """Compute the full restore path from restore argument.

    Args:
        restore_arg: Either a Path to a checkpoint, or "default" to use default_path.
        default_path: Default path to use when restore_arg is "default".
        gcs_bucket: GCS bucket name. When set, paths are prefixed with gs://{bucket}/.

    Returns:
        The full restore path as a string.
    """
    if restore_arg == "default":
        if gcs_bucket is not None:
            return f"gs://{gcs_bucket}/{default_path}"
        return default_path
    elif gcs_bucket is not None:
        return f"gs://{gcs_bucket}/{restore_arg}"
    else:
        return str(restore_arg.absolute())


def open_restore_manager(
    restore_arg: Path | Literal["default"] | None,
    maybe_restore_arg: Path | Literal["default"] | None,
    default_path: str,
    gcs_bucket: str | None,
    item_names: list[str],
) -> tuple[ocp.CheckpointManager | None, int]:
    """Open a checkpoint manager for restoring, with maybe-restore support.

    Args:
        restore_arg: Value of --restore argument (crashes if no checkpoint found).
        maybe_restore_arg: Value of --maybe-restore argument (warns if no checkpoint found).
        default_path: Default path to use when restore_arg is "default".
        gcs_bucket: GCS bucket name for path prefixing.
        item_names: List of item names for the checkpoint manager.

    Returns:
        Tuple of (restore_manager, step). If no checkpoint is found and maybe_restore
        mode is active, returns (None, 0). If no restore is requested, returns (None, 0).

    Raises:
        ValueError: If --restore is used and no checkpoint is found.
        ValueError: If both --restore and --maybe-restore are provided.
    """
    if restore_arg is not None and maybe_restore_arg is not None:
        raise ValueError("--restore and --maybe-restore are mutually exclusive")

    # Determine which restore argument to use
    effective_arg = restore_arg if restore_arg is not None else maybe_restore_arg
    is_maybe_restore = restore_arg is None and maybe_restore_arg is not None

    if effective_arg is None:
        return None, 0

    restore_path = compute_restore_path(effective_arg, default_path, gcs_bucket)
    opts = ocp.CheckpointManagerOptions(read_only=True)
    restore_mngr = ocp.CheckpointManager(restore_path, options=opts, item_names=item_names)
    step = restore_mngr.latest_step()

    if step is None:
        if is_maybe_restore:
            logging.warning(f"No checkpoint found at {restore_path}. Training from scratch.")
            return None, 0
        else:
            raise ValueError(f"No checkpoint found in path {restore_path}")

    logging.info(f"Found checkpoint at step {step}. Restoring ...")
    return restore_mngr, step


def restore_optimizer_state(
    restore_mngr: ocp.CheckpointManager,
    step: int,
    optim: nnx.Optimizer,
    mesh: Mesh,
    name: str = "optim",
) -> None:
    """Restore optimizer state from checkpoint.

    Args:
        restore_mngr: Checkpoint manager to restore from.
        step: Checkpoint step to restore.
        optim: Optimizer to restore state into.
        mesh: JAX mesh for sharding.
        name: Name of the optimizer item in the checkpoint.
    """
    optim_state = nnx.state(optim)
    optim_pure_state = nnx.to_pure_dict(optim_state)
    optim_restore_args = jax.tree.map(
        lambda x: ocp.ArrayRestoreArgs(sharding=NamedSharding(mesh, P())), optim_pure_state
    )
    optim_pure_state = restore_mngr.restore(
        step,
        args=ocp.args.Composite(
            **{name: ocp.args.PyTreeRestore(optim_pure_state, restore_args=optim_restore_args)}
        ),
    )[name]
    nnx.replace_by_pure_dict(optim_state, optim_pure_state)
    nnx.update(optim, optim_state)


def restore_data_loader(
    restore_mngr: ocp.CheckpointManager,
    step: int,
    data_iter: grain.DataLoader,
    name: str = "loader",
    graceful: bool = True,
) -> grain.DataLoader:
    """Restore data loader state from checkpoint.

    Args:
        restore_mngr: Checkpoint manager to restore from.
        step: Checkpoint step to restore.
        data_iter: Data iterator to restore state into.
        name: Name of the loader item in the checkpoint.
        graceful: If True, log error and return original data_iter on failure.
            If False, raise the exception.

    Returns:
        The restored data iterator, or original on failure if graceful=True.
    """
    try:
        return restore_mngr.restore(
            step, args=ocp.args.Composite(**{name: grain.PyGrainCheckpointRestore(data_iter)})
        )[name]
    except Exception as e:
        if graceful:
            logging.error(
                f"Failed to restore data loader checkpoint (e.g., worker count mismatch). "
                f"Starting data iteration from scratch. Error: {e}"
            )
            return data_iter
        raise


def restore_model_state(
    restore_mngr: ocp.CheckpointManager,
    step: int,
    model: nnx.Module,
    mesh: Mesh,
    name: str,
) -> None:
    """Restore model state from checkpoint.

    Args:
        restore_mngr: Checkpoint manager to restore from.
        step: Checkpoint step to restore.
        model: Model to restore state into.
        mesh: JAX mesh for sharding.
        name: Name of the model item in the checkpoint.
    """
    state = nnx.state(model)
    pure_state = nnx.to_pure_dict(state)
    restore_args = jax.tree.map(
        lambda x: ocp.ArrayRestoreArgs(sharding=NamedSharding(mesh, P())), pure_state
    )
    restored_state = restore_mngr.restore(
        step,
        args=ocp.args.Composite(
            **{name: ocp.args.PyTreeRestore(pure_state, restore_args=restore_args)}
        ),
    )[name]
    nnx.replace_by_pure_dict(state, restored_state)
    nnx.update(model, state)


# =============================================================================
# Statistics Utilities
# =============================================================================


def update_running_stats(
    count: int,
    mean: jax.Array,
    m2: jax.Array,
    batch: jax.Array,
) -> tuple[int, jax.Array, jax.Array]:
    """Update running mean and M2 (sum of squared deviations) using Welford's parallel algorithm.

    This is numerically stable for large datasets. M2 tracks the sum of squared
    deviations from the mean, from which variance can be computed as M2 / count.

    Reference:
        Chan, T.F., Golub, G.H., LeVeque, R.J. (1979). "Updating Formulae and a
        Pairwise Algorithm for Computing Sample Variances."

    Args:
        count: Current sample count.
        mean: Current running mean, shape (T, D).
        m2: Current M2 (sum of squared deviations), shape (T, D).
        batch: New batch of data, shape (B, T, D).

    Returns:
        Updated (count, mean, m2).

    Example:
        >>> # Initialize accumulators
        >>> count = 0
        >>> mean = jnp.zeros((num_tokens, embed_dim), dtype=jnp.float64)
        >>> m2 = jnp.zeros((num_tokens, embed_dim), dtype=jnp.float64)
        >>>
        >>> # Update with batches
        >>> for batch in dataloader:
        >>>     count, mean, m2 = update_running_stats(count, mean, m2, batch)
        >>>
        >>> # Compute final variance
        >>> variance = m2 / count
    """
    batch_count = batch.shape[0]
    batch_mean = jnp.mean(batch, axis=0)  # (T, D)
    batch_var = jnp.var(batch, axis=0)  # (T, D) - population variance
    batch_m2 = batch_var * batch_count

    # Parallel combination formula
    new_count = count + batch_count
    delta = batch_mean - mean
    new_mean = mean + delta * batch_count / new_count
    new_m2 = m2 + batch_m2 + delta**2 * count * batch_count / new_count

    return new_count, new_mean, new_m2
