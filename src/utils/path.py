from pathlib import Path


FILE_PATH = Path(__file__).resolve()
UTILS_DIR = FILE_PATH.parent
SRC_DIR = UTILS_DIR.parent
ROOT_DIR = SRC_DIR.parent


DATA_DIR = ROOT_DIR / "data"


MODELS_DIR = ROOT_DIR / "models"


OUT_DIR = ROOT_DIR / "output"
