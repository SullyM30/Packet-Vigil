"""
API configuration constants and settings.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# API settings
API_HOST = '0.0.0.0'
API_PORT = 5000
DEBUG_MODE = False

# Secret key for Flask sessions 
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Dashboard settings
DASHBOARD_REFRESH_INTERVAL = 5  # seconds
MAX_ALERTS_DISPLAY = 1000

# Detection settings
DEFAULT_AE_THRESHOLD = 0.0198
DEFAULT_RF_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 1000
DEFAULT_DETECTION_INTERVAL = 1.0  # seconds

# Alert rate status thresholds
ALERT_RATE_NORMAL = 1.0    # alerts per second
ALERT_RATE_ELEVATED = 5.0  # alerts per second

# Logging
LOG_FILE = LOGS_DIR / "app.log"
LOG_LEVEL = "INFO"

# Development mode - set to True to run without trained models (uses mock data)
MOCK_MODE = os.environ.get('MOCK_MODE', 'False').lower() == 'true'

# External detection mode — set to True when replay_detector.py or
# live_detector_linux.py is feeding alerts/metrics directly into UI/data/.
# Disables the built-in detection engine so it doesn't overwrite live data.
EXTERNAL_DETECTION_MODE = os.environ.get('EXTERNAL_DETECTION_MODE', 'True').lower() == 'true'
