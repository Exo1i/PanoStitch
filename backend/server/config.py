"""
Configuration for PanoStitch Backend
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Flask Configuration
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
DEBUG = os.getenv('DEBUG', 'True') == 'True'
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 5000))

# File Upload Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'}

# Stitching Defaults
DEFAULT_RESIZE = 800
DEFAULT_FOCAL_LENGTH = 1200.0
DEFAULT_USE_DNN = False
DEFAULT_USE_HARRIS = False
DEFAULT_USE_CYLINDRICAL = True

# API Configuration
API_VERSION = '1.0.0'
API_TITLE = 'PanoStitch Backend API'
API_BASE_URL = '/api'

# Directories
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER', 'results')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
