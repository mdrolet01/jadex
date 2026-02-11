from pathlib import Path

# This should not be changed
JADEX_ROOT_DIR = Path(__file__).resolve().parent.parent

# User configurable
CACHE_DIR = Path.home() / ".cache" / "jadex"
JADEX_CHECKPOINT_DIR = JADEX_ROOT_DIR / "checkpoints"
