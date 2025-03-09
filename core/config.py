import os
import logging

# Create log -----------------------------------
# Create Logs folder
LOG_DIR  = "Logs"
os.makedirs(LOG_DIR , exist_ok=True)

# Set log file path
LOG_FILE  = os.path.join(LOG_DIR , "app.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE ,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

# Logger instance (use this across files)
logger=logging.getLogger(__name__)

# Create Models folder -----------------------------------
MODEL_DIR = "Machine_Learning_Models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_REGISTRY_FILE = "model_registry.csv"