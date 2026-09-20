"""Backward-compatible command-line entry point for METABRIC training."""

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from metabric.training import (  # noqa: E402,F401
    bootstrap_auc,
    build_pipeline,
    load_raw,
    main,
    prepare_features,
)


if __name__ == "__main__":
    main()
