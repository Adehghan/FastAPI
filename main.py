# import os
from stat import filemode
from fastapi import FastAPI
# import logging
from app.api import users_get
from app.api import users_post
from app.api import cnn
from core.config import logger, LOG_DIR, MODEL_DIR

app = FastAPI()
app.include_router(users_get.router)
app.include_router(users_post.router)
app.include_router(cnn.router)

# # Create log -----------------------------------

# # Step 1: Create "Logs" folder if it doesn't exist
# log_folder = "Logs"
# os.makedirs(log_folder, exist_ok=True)

# # Step 2: Configure logging to write to Logs/app.log
# log_file_path = os.path.join(log_folder, "app.log")

# logging.basicConfig(
#     level=logging.INFO,
#     filename=log_file_path,
#     filemode='a',
#     format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
# )

# # Step 3: Create logger instance
# logger=logging.getLogger(__name__)

# # Create Directory for saved models -----------------------------------
# MODEL_DIR = "Models"
# os.makedirs(MODEL_DIR, exist_ok=True)




