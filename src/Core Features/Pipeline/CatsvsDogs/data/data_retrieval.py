import sys
import tempfile
import traceback
from pathlib import Path
import shutil

ORIGINAL_DATA: Path = Path(
    r"C:\Users\natte\Documents\Projects\ClearML-Demo\src\Core Features\Pipeline\CatsvsDogs\data\pictures"
)

try:
    # Create temporary folders for the Train, Val, Test Split of the
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path: Path = Path(tmp_dir.name)

    train_dir: Path = tmp_path / "train"
    train_dir.mkdir()

    val_dir: Path = tmp_path / "val"
    val_dir.mkdir()

    test_dir: Path = tmp_path / "test"
    test_dir.mkdir()

    # Subdirectories for cats and dogs
    # Trainings Data dirs
    train_cats_dir: Path = train_dir / "cats"
    train_cats_dir.mkdir()
    train_dogs_dir: Path = train_dir / "dogs"
    train_dogs_dir.mkdir()
    # Validation Data dirs
    val_cats_dir: Path = val_dir / "cats"
    val_cats_dir.mkdir()
    val_dogs_dir: Path = val_dir / "dogs"
    val_dogs_dir.mkdir()
    # Test Data dirs
    test_cats_dir: Path = test_dir / "cats"
    test_cats_dir.mkdir()
    test_dogs_dir: Path = test_dir / "dogs"
    test_dogs_dir.mkdir()

    # Copy pictures from the source directory into the create dirs,
    # following a .5/.25/.25 split between Training, Validation and Testing Data
    # amounting to a total of 2.000 pictures per cats and per dogs

    # Cats
    fnames: list[str] = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src: Path = ORIGINAL_DATA / fname
        dst: Path = train_cats_dir / fname
        shutil.copyfile(src, dst)
    fnames: list[str] = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src: Path = ORIGINAL_DATA / fname
        dst: Path = val_cats_dir / fname
        shutil.copyfile(src, dst)
    fnames: list[str] = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src: Path = ORIGINAL_DATA / fname
        dst: Path = test_cats_dir / fname
        shutil.copyfile(src, dst)

    # Dogs
    fnames: list[str] = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src: Path = ORIGINAL_DATA / fname
        dst: Path = train_dogs_dir / fname
        shutil.copyfile(src, dst)
    fnames: list[str] = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src: Path = ORIGINAL_DATA / fname
        dst: Path = val_dogs_dir / fname
        shutil.copyfile(src, dst)
    fnames: list[str] = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src: Path = ORIGINAL_DATA / fname
        dst: Path = test_do / fname
        shutil.copyfile(src, dst)

    cats





except Exception:
    exc_info = sys.exc_info()
    tmp_dir.cleanup()

finally:
    if 'exc_info' in locals():
        traceback.print_exception(exc_info)
        del exc_info
