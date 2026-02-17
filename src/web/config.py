import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent

MODEL_PATH = BASE_DIR / 'saved_models' / 'kan_cnn.pt'

NUM_SAMPLES = 25

IMAGE_SIZE = (28, 28)

COLOR_MIN = '#001a00'
COLOR_MAX = '#00ff00'

FLASK_DEBUG = True
FLASK_PORT = 5001
FLASK_HOST = '0.0.0.0'

STATIC_FOLDER = BASE_DIR / 'src' / 'web' / 'static'
TEMPLATES_FOLDER = BASE_DIR / 'src' / 'web' / 'templates'
SAMPLES_FOLDER = STATIC_FOLDER / 'samples'
UPLOAD_FOLDER = BASE_DIR / 'saved_models'

ALLOWED_MODEL_EXTENSIONS = {'.pt', '.pth'}

os.makedirs(SAMPLES_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
