import sys
import tempfile
import traceback
from pathlib import Path

# TODO DataLocation
ORIGINAL_DATA: str = "TODO"

try:
    # Create temporary folders for the Train, Val, Test Split of the
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path: Path = Path(tmp_dir.name)
    train_dir: Path = tmp_path / "train"
    train_dir.mkdir()
    val_dir: Path = tmp_path / "val"
    val_dir.mkdir()
    test_dir: Path = tmp_path / "val"


except Exception:
    exc_info = sys.exc_info()
    tmp_dir.cleanup()

finally:
    if 'exc_info' in locals():
        traceback.print_exception(exc_info)
        del exc_info
