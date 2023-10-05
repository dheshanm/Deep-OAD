import os
import subprocess


def get_repo_root() -> str:
    """
    Returns the root directory of the current Git repository.

    Uses the command `git rev-parse --show-toplevel` to get the root directory.
    """
    repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    repo_root = repo_root.decode("utf-8").strip()

    repo_root = os.path.join(repo_root, "deep_oad")

    return repo_root


ROOT_DIR = get_repo_root()
VIT_WEIGHTS_PATH = os.path.join(ROOT_DIR, "weights", "model-vit-ang-loss.h5")

SAVE_IMAGE_DIR = os.path.join("/tmp/")

COCO_TRAIN_DIR = "data/train/"
COCO_VALIDATION_DIR = "data/validation-test"
COCO_VALIDATION_TEST_LABEL_CSV_PATH = "data/validation-test.csv"
BATCH_SIZE = 16
