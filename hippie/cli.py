"""Command-line entry point for the HIPPIE package (``hippie-cli``).

Self-contained: every subcommand operates on the installed ``hippie`` package
only and performs no network access. Subcommands:

    hippie-cli version       Print the installed package version.
    hippie-cli info          Print package info and benchmark defaults.
    hippie-cli configs       List the predefined model configurations.
    hippie-cli smoke-test    Build a model and run one forward pass on
                             synthetic input to verify the install works.
"""

from __future__ import annotations

import argparse
import sys


def _get_version() -> str:
    try:
        from importlib.metadata import version

        return version("hippie")
    except Exception:
        return "0.1.0"


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"hippie {_get_version()}")
    return 0


def _cmd_info(_args: argparse.Namespace) -> int:
    print(f"HIPPIE {_get_version()}")
    print("Conditional convolutional joint autoencoder for extracellular")
    print("electrophysiology (waveform / ISI / ACG).")
    print()
    print("Benchmark defaults:  beta=1.0  z_dim=30 (10 per modality)")
    print("Generative defaults: beta=0.1  z_dim=16")
    print("Modality input sizes: wave=50  isi=100  acg=100")
    print()
    print("Training entry points live in scripts/ (run from the repo root).")
    print("Pretrained checkpoint: https://huggingface.co/Jesusgf23/hippie")
    return 0


def _config_names() -> list[str]:
    from .multimodal_model import ExperimentConfigs

    return [
        name
        for name in dir(ExperimentConfigs)
        if not name.startswith("_")
        and callable(getattr(ExperimentConfigs, name))
    ]


def _cmd_configs(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from .multimodal_model import ExperimentConfigs

    names = _config_names()

    if args.name:
        if args.name not in names:
            print(f"Unknown config '{args.name}'. Available:", file=sys.stderr)
            for name in names:
                print(f"  {name}", file=sys.stderr)
            return 1
        cfg = getattr(ExperimentConfigs, args.name)()
        print(f"{args.name}:")
        for key, value in asdict(cfg).items():
            print(f"  {key} = {value}")
        return 0

    print("Available model configurations:")
    for name in names:
        doc = (getattr(ExperimentConfigs, name).__doc__ or "").strip().splitlines()
        summary = doc[0] if doc else ""
        print(f"  {name:<32} {summary}")
    print()
    print("Use 'hippie-cli configs --name <NAME>' to see all fields.")
    return 0


def _cmd_smoke_test(_args: argparse.Namespace) -> int:
    import torch

    from .multimodal_model import ExperimentConfigs, MultiModalCVAE

    modalities = {"wave": 50, "isi": 100, "acg": 100}
    z_dim = 30
    num_sources, num_classes = 3, 5

    model = MultiModalCVAE(
        modalities=modalities,
        z_dim=z_dim,
        config=ExperimentConfigs.class_decoder_source_bn_aug_reg(),
        num_sources=num_sources,
        num_classes=num_classes,
    )
    model.eval()

    batch = {name: torch.zeros(2, 1, size) for name, size in modalities.items()}
    source_labels = torch.zeros(2, dtype=torch.long)

    with torch.no_grad():
        # class_labels=None masks the class embedding, matching evaluation-time behavior.
        _, z_mean, _ = model.encode(
            batch, source_labels=source_labels, class_labels=None
        )

    if not torch.isfinite(z_mean).all():
        print("[smoke-test] FAILED: non-finite latent embedding.", file=sys.stderr)
        return 1

    print(f"[smoke-test] OK — encoded latent shape {tuple(z_mean.shape)} (z_dim={z_dim}).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hippie-cli", description="HIPPIE command-line interface."
    )
    parser.add_argument(
        "--version", action="version", version=f"hippie {_get_version()}"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="Print the installed package version.")
    sub.add_parser("info", help="Print package info and benchmark defaults.")

    configs_parser = sub.add_parser(
        "configs", help="List predefined model configurations."
    )
    configs_parser.add_argument(
        "--name", help="Show all fields for a single named configuration."
    )

    sub.add_parser(
        "smoke-test",
        help="Build a model and run one forward pass on synthetic input.",
    )

    args = parser.parse_args(argv)

    handlers = {
        "version": _cmd_version,
        "info": _cmd_info,
        "configs": _cmd_configs,
        "smoke-test": _cmd_smoke_test,
    }
    if args.command is None:
        parser.print_help()
        return 0
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
