#!/usr/bin/env python3
"""Benchmark DiT vs FlatDINO forward pass performance.

Measures forward passes per second for DiT-XL, DiT-XL DH, and FlatDINO DiT-XL
across different batch sizes. Results are saved to a JSON file and can be
plotted per device.

Example usage:
    # Run benchmarks on current device
    python -m flatdino.eval.performance_gpu --device-name "A100-80GB"

    # Generate plots from existing results
    python -m flatdino.eval.performance_gpu --plot

    # Custom number of iterations
    python -m flatdino.eval.performance_gpu --device-name "H100" --num-iters 30
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import jmp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from absl import logging
from flax import nnx
from jax.sharding import Mesh

from flatdino.models.dit import LightningDiT, DiTConfig, LIGHTNING_DIT_CONFIGS
from flatdino.models.transformer import TransformerConfig
from flatdino.decoder.dit_dh import DiTDH, DiTDHConfig

# Configure matplotlib for publication-quality figures
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["STIXGeneral"]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.labelsize"] = 22
mpl.rcParams["axes.titlesize"] = 24
mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20
mpl.rcParams["legend.fontsize"] = 18
mpl.rcParams["figure.dpi"] = 150

# Color palette (from scripts/utils.py)
CONTRAST_PALETTE = [
    "#3B9AB2",  # Teal
    "#F21A00",  # Red
    "#00A08A",  # Green
    "#F98400",  # Orange
    "#5BBCD6",  # Light blue
]

DEFAULT_JSON_PATH = Path("./profiling.json")
BATCH_SIZES = [2**i for i in range(11)]  # 1, 2, 4, ..., 1024

# Model configurations
DIT_XL_CONFIG = LIGHTNING_DIT_CONFIGS["dit-xl"]


@dataclass
class ModelSpec:
    """Specification for a model to benchmark."""

    name: str
    num_tokens: int
    in_channels: int
    model_type: str  # "dit" or "dit_dh"


MODEL_SPECS = [
    ModelSpec("DiT-XL (256 tokens)", num_tokens=256, in_channels=768, model_type="dit"),
    ModelSpec("DiT-XL DH (256 tokens)", num_tokens=256, in_channels=768, model_type="dit_dh"),
    ModelSpec("FlatDINO DiT-XL (32 tokens)", num_tokens=32, in_channels=768, model_type="dit"),
]


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single model at a single batch size."""

    model_name: str
    batch_size: int
    forward_passes_per_second: float
    avg_time_ms: float


@dataclass
class DeviceResults:
    """Results for all models on a single device."""

    device_name: str
    results: dict[str, dict[int, float]] = field(default_factory=dict)
    # results[model_name][batch_size] = forward_passes_per_second
    oom_batch_sizes: dict[str, int | None] = field(default_factory=dict)
    # oom_batch_sizes[model_name] = first batch size that OOMed (or None if no OOM)


def create_model(spec: ModelSpec, mp: jmp.Policy, rngs: nnx.Rngs):
    """Create a model instance from specification."""
    if spec.model_type == "dit":
        cfg = DiTConfig(
            transformer=TransformerConfig(**DIT_XL_CONFIG),
            in_channels=spec.in_channels,
            num_patches=spec.num_tokens,
            patch_embed=None,
        )
        return LightningDiT(cfg, mp, rngs=rngs, num_tokens=spec.num_tokens)
    elif spec.model_type == "dit_dh":
        cfg = DiTDHConfig(
            input_size=spec.num_tokens,
            patch_size=None,
            in_channels=spec.in_channels,
            encoder_dim=DIT_XL_CONFIG["embed_dim"],
            encoder_heads=DIT_XL_CONFIG["num_heads"],
            encoder_depth=DIT_XL_CONFIG["num_layers"],
            mlp_ratio=DIT_XL_CONFIG["mlp_hidden_dim"] / DIT_XL_CONFIG["embed_dim"],
        )
        return DiTDH(cfg, mp, rngs=rngs)
    else:
        raise ValueError(f"Unknown model type: {spec.model_type}")


def create_forward_fn(model):
    """Create a JIT-compiled forward function for benchmarking.

    Uses nnx.split to separate model into graphdef and state.
    Returns a JIT function that takes (state, x, t, labels) as arguments
    to avoid capturing large constants and potential memory duplication.
    """
    graphdef, state = nnx.split(model)

    @jax.jit
    def forward_fn(state, x, t, labels):
        model = nnx.merge(graphdef, state)
        return model(x, t, labels, train=False)

    return forward_fn, state


def _time_with_mosaic(run_forward, num_warmup: int, num_iters: int) -> list[float]:
    """Time using mosaic GPU profiler (CUPTI-based, high accuracy).

    Returns runtime in milliseconds.
    """
    from jax.experimental.mosaic.gpu import profiler

    # Warmup runs (without timing)
    warmup_fn = profiler.measure(run_forward, aggregate=True, iterations=num_warmup)
    warmup_fn()

    # Timed runs - use iterations parameter for efficiency
    measured_fn = profiler.measure(run_forward, aggregate=True, iterations=num_iters)
    _, timings = measured_fn()

    # timings is a list of measurements when iterations > 1
    if timings is None:
        return [0.0] * num_iters
    elif isinstance(timings, list):
        return timings
    else:
        # Single measurement (shouldn't happen with iterations > 1)
        return [timings]


def _time_with_block(run_forward, num_warmup: int, num_iters: int) -> list[float]:
    """Time using jax.block_until_ready (fallback)."""
    import time

    # Warmup runs
    for _ in range(num_warmup):
        result = run_forward()
        jax.block_until_ready(result)

    # Timed runs
    times_ms = []
    for _ in range(num_iters):
        start = time.perf_counter()
        result = run_forward()
        jax.block_until_ready(result)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)

    return times_ms


def benchmark_single_batch(
    spec: ModelSpec,
    batch_size: int,
    mp: jmp.Policy,
    rngs: nnx.Rngs,
    timing_fn,
    num_warmup: int,
    num_iters: int,
) -> tuple[BenchmarkResult | None, bool]:
    """Benchmark a single model at a single batch size.

    Returns:
        Tuple of (result, is_oom) where result is None if an error occurred,
        and is_oom indicates whether the error was an OOM.
    """
    logging.info(f"  Benchmarking {spec.name} with batch_size={batch_size}")

    # Create fresh model for each batch size to avoid memory issues
    model = create_model(spec, mp, rngs)

    # Create JIT-compiled forward function (state passed as argument, not constant)
    forward_fn, state = create_forward_fn(model)

    # Create inputs
    x = jnp.zeros((batch_size, spec.num_tokens, spec.in_channels), dtype=jnp.bfloat16)
    t = jax.random.uniform(jax.random.key(0), (batch_size,), dtype=jnp.bfloat16)
    labels = jnp.zeros((batch_size,), dtype=jnp.int32)

    # Compile the function by running once (outside profiler)
    # This ensures compilation doesn't affect timing
    _ = forward_fn(state, x, t, labels)

    # Create no-arg wrapper for profiler (inputs are small, only ~batch_size floats)
    def run_forward():
        return forward_fn(state, x, t, labels)

    result = None
    is_oom = False

    try:
        times_ms = timing_fn(run_forward, num_warmup, num_iters)

        avg_time_ms = np.mean(times_ms)
        std_time_ms = np.std(times_ms)
        fps = 1000.0 / avg_time_ms  # Forward passes per second

        result = BenchmarkResult(
            model_name=spec.name,
            batch_size=batch_size,
            forward_passes_per_second=fps,
            avg_time_ms=avg_time_ms,
        )
        logging.info(
            f"    -> {fps:.2f} forward passes/sec ({avg_time_ms:.2f} +/- {std_time_ms:.2f} ms/pass)"
        )

    except Exception as e:
        error_str = str(e).lower()
        if "out of memory" in error_str or "oom" in error_str or "resource exhausted" in error_str:
            logging.warning(
                f"    -> OOM at batch_size={batch_size}. "
                f"Skipping larger batch sizes for {spec.name}."
            )
            is_oom = True
        else:
            logging.warning(f"    -> Failed: {e}")

    # Clean up
    del model, forward_fn, state, x, t, labels
    jax.clear_caches()

    return result, is_oom


def load_results(json_path: Path) -> dict[str, DeviceResults]:
    """Load existing results from JSON file."""
    if not json_path.exists():
        return {}

    with open(json_path) as f:
        data = json.load(f)

    results = {}
    for device_name, device_data in data.items():
        dr = DeviceResults(device_name=device_name)
        # Handle both old format (just results) and new format (results + oom)
        if "results" in device_data:
            # New format
            for model_name, batch_results in device_data["results"].items():
                dr.results[model_name] = {int(k): v for k, v in batch_results.items()}
            for model_name, oom_bs in device_data.get("oom_batch_sizes", {}).items():
                dr.oom_batch_sizes[model_name] = oom_bs
        else:
            # Old format (backwards compatibility)
            for model_name, batch_results in device_data.items():
                dr.results[model_name] = {int(k): v for k, v in batch_results.items()}
        results[device_name] = dr

    return results


def save_results(json_path: Path, all_results: dict[str, DeviceResults]):
    """Save results to JSON file."""
    data = {}
    for device_name, dr in all_results.items():
        data[device_name] = {
            "results": dr.results,
            "oom_batch_sizes": dr.oom_batch_sizes,
        }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    logging.info(f"Saved results to {json_path}")


def run_benchmarks(
    device_name: str,
    json_path: Path,
    num_warmup: int = 3,
    num_iters: int = 15,
    use_mosaic: bool = True,
    skip: bool = False,
):
    """Run benchmarks for all models and save results.

    Results are saved to JSON after each successful benchmark, providing
    fault tolerance in case of crashes or interruptions.
    """
    logging.info(f"Running benchmarks on device: {device_name}")

    # Load existing results
    all_results = load_results(json_path)

    # Get or create device results
    if device_name in all_results:
        device_results = all_results[device_name]
    else:
        device_results = DeviceResults(device_name=device_name)

    # Set up shared state for benchmarking
    rngs = nnx.Rngs(0)
    mp = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        output_dtype=jnp.float32,
    )

    # Create a simple mesh for single-GPU benchmarking (required for FSDP annotations)
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = Mesh(devices, ("data", "model"))
    jax.set_mesh(mesh)

    # Select timing function
    if use_mosaic:
        try:
            timing_fn = _time_with_mosaic
            logging.info("Using mosaic GPU profiler for timing")
        except ImportError:
            logging.warning("Mosaic profiler not available, falling back to block_until_ready")
            timing_fn = _time_with_block
    else:
        timing_fn = _time_with_block
        logging.info("Using block_until_ready for timing")

    for spec in MODEL_SPECS:
        logging.info(f"Benchmarking {spec.name}...")

        # Get existing results for this model
        existing_batch_sizes = set(device_results.results.get(spec.name, {}).keys())
        existing_oom = device_results.oom_batch_sizes.get(spec.name)

        # Initialize results dict for this model if needed
        if spec.name not in device_results.results:
            device_results.results[spec.name] = {}

        for batch_size in BATCH_SIZES:
            # Skip if we already have this result
            if skip and batch_size in existing_batch_sizes:
                logging.info(f"  Skipping batch_size={batch_size} (already benchmarked)")
                continue

            # Skip if we already hit OOM at this or smaller batch size
            if existing_oom is not None and batch_size >= existing_oom:
                logging.info(f"  Skipping batch_size={batch_size} (OOM at {existing_oom})")
                continue

            result, is_oom = benchmark_single_batch(
                spec, batch_size, mp, rngs, timing_fn, num_warmup, num_iters
            )

            if result is not None:
                # Store the result
                device_results.results[spec.name][batch_size] = result.forward_passes_per_second

                # Update all_results and save immediately
                all_results[device_name] = device_results
                save_results(json_path, all_results)

            if is_oom:
                # Record OOM and stop benchmarking larger batch sizes for this model
                device_results.oom_batch_sizes[spec.name] = batch_size
                all_results[device_name] = device_results
                save_results(json_path, all_results)
                break


def plot_results(json_path: Path, output_dir: Path, title: str | None = None):
    """Generate performance plots from JSON results."""
    all_results = load_results(json_path)

    if not all_results:
        logging.error(f"No results found in {json_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for device_name, device_results in all_results.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        # Track y-axis minimum for placing OOM markers
        all_fps_values = []

        for i, (model_name, batch_data) in enumerate(device_results.results.items()):
            if not batch_data:
                continue

            batch_sizes = sorted(batch_data.keys())
            fps_values = [batch_data[bs] for bs in batch_sizes]
            all_fps_values.extend(fps_values)

            color = CONTRAST_PALETTE[i % len(CONTRAST_PALETTE)]
            ax.plot(
                batch_sizes,
                fps_values,
                marker="o",
                markersize=8,
                linewidth=2,
                color=color,
                label=model_name,
            )

            # Add X marker at OOM batch size if applicable
            oom_bs = device_results.oom_batch_sizes.get(model_name)
            if oom_bs is not None:
                # Place X marker at the OOM batch size, at the bottom of the plot
                # We'll set the y position after we know the axis limits
                ax.axvline(x=oom_bs, color=color, linestyle=":", alpha=0.5, linewidth=1.5)
                # Store for later annotation
                if not hasattr(ax, "_oom_markers"):
                    ax._oom_markers = []
                ax._oom_markers.append((oom_bs, color, model_name))

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")

        # Set x-axis ticks to powers of 2
        ax.set_xticks(BATCH_SIZES)
        ax.set_xticklabels([str(bs) for bs in BATCH_SIZES])
        ax.minorticks_off()

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Forward Passes / Second")
        if title is not None:
            ax.set_title(title)

        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add X markers at OOM points (after axis limits are set)
        if hasattr(ax, "_oom_markers") and ax._oom_markers:
            y_min, y_max = ax.get_ylim()
            # Place X markers at the bottom of the plot
            oom_y = y_min * 1.5  # Slightly above the bottom
            for oom_bs, color, model_name in ax._oom_markers:
                ax.scatter(
                    [oom_bs],
                    [oom_y],
                    marker="x",
                    s=150,
                    color=color,
                    linewidths=3,
                    zorder=10,
                )
                ax.annotate(
                    "OOM",
                    (oom_bs, oom_y),
                    textcoords="offset points",
                    xytext=(0, -15),
                    ha="center",
                    fontsize=10,
                    color=color,
                    fontweight="bold",
                )

        # Save
        safe_device_name = device_name.replace(" ", "_").replace("/", "-")
        output_path = output_dir / f"performance_{safe_device_name}"
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        logging.info(f"Saved plot to {output_path.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DiT vs FlatDINO forward pass performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--device-name",
        type=str,
        help="Name of the device being benchmarked (e.g., 'A100-80GB', 'H100')",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help=f"Path to JSON file for storing results (default: {DEFAULT_JSON_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./plots"),
        help="Directory for output plots (default: ./plots)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=3,
        help="Number of warmup iterations before timing (default: 3)",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=15,
        help="Number of iterations to average for timing (default: 15)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Only generate plots from existing JSON results (no benchmarking)",
    )
    parser.add_argument(
        "--no-mosaic",
        action="store_true",
        help="Use simple timing with jax.block_until_ready instead of mosaic profiler",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title for the plot (default: no title)",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Skip benchmarking if device already has results in the JSON file",
    )
    args = parser.parse_args()

    logging.set_verbosity(logging.INFO)

    if args.plot:
        # Just generate plots
        plot_results(args.json_path, args.output_dir, title=args.title)
    else:
        # Run benchmarks
        if args.device_name is None:
            parser.error("--device-name is required when running benchmarks")

        run_benchmarks(
            device_name=args.device_name,
            json_path=args.json_path,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
            use_mosaic=not args.no_mosaic,
            skip=args.skip,
        )

        # Also generate plots after benchmarking
        plot_results(args.json_path, args.output_dir, title=args.title)


if __name__ == "__main__":
    main()
