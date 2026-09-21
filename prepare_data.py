"""Command-line entry point: regenerate data/processed from the raw METABRIC file."""

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from metabric.prepare_data import main  # noqa: E402


if __name__ == "__main__":
    main()
