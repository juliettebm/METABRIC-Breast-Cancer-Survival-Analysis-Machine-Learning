"""Command-line entry point for the METABRIC robustness diagnostics."""

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from metabric.robustness import main  # noqa: E402


if __name__ == "__main__":
    main()
