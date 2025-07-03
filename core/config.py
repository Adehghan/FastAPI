from logging.handlers import TimedRotatingFileHandler
import os
import logging


# Get the absolute path to the project root (assuming config.py is in core/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Build absolute paths
MODEL_DIR = os.path.join(PROJECT_ROOT, "Machine_Learning_Models")
MODEL_REGISTRY_FILE = os.path.join(PROJECT_ROOT, "model_registry.csv")

# Log file (optional, same idea)
LOG_DIR = os.path.join(PROJECT_ROOT, "Logs")
# LOG_FILE = os.path.join(LOG_DIR, "app.log")
from datetime import datetime
today_str = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = os.path.join(LOG_DIR, f"app_{today_str}.log")

# Make sure folders exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     filename=LOG_FILE ,
#     filemode='a',
#     format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
# )

# # Logger instance (use this across files)
# logger=logging.getLogger(__name__)

# Create handler with daily rotation
handler = TimedRotatingFileHandler(
    LOG_FILE,
    when='midnight',       # Rotate at midnight
    interval=1,            # Every 1 day
    backupCount=30,        # Keep last 30 days
    encoding='utf-8',
    utc=False              # Set to True if your server uses UTC
)

# Optional: custom filename suffix
#handler.suffix = "%Y-%m-%d"

# Formatter
formatter = logging.Formatter('%(asctime)s -- %(levelname)s -- %(name)s -- %(message)s')
handler.setFormatter(formatter)

# Configure root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False  # Optional: avoid duplicate logs if using FastAPI/uvicorn