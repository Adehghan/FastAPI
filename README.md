# FastAPI ML Model Management and Serving
This project provides a FastAPI-based microservice for training, registering, and serving machine learning models. It allows saving the best-performing models, logging the training process, and exposing RESTful endpoints for predictions.

---

## Features
- Model Registry: Automatically saves and registers trained ML models with their performance metrics in model_registry.csv.
- Model Selection: Provides the ability to fetch and load the best model based on historical evaluation results.
- Logging: Records training and prediction activities in structured log files under the Logs/ directory.
- Modular Structure: Cleanly separated components in app/api/, app/core/, and DataAccess/ directories.
- Flexible Data Access: Organized datasets and models under DataAccess/ for easy management.

---

## ML Capabilities
- Includes training and inference on various NLP and computer vision tasks, such as:
- Sentiment analysis on Amazon food reviews
- Flower species recognition
- Named entity recognition
- Regression models
- 
---

## Technologies Used
- FastAPI for API design
- Python & scikit-learn for ML pipelines
- Pandas & NumPy for data manipulation
- Custom logging module for process traceability

---

## Getting Started
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/fastapi-ml-service.git
cd fastapi-ml-service
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the application locally:

bash
Copy
Edit
uvicorn main:app --reload
Access the interactive API docs at http://127.0.0.1:8000/docs

---

## Future Work
- Add JWT-based authentication for secure API access.
- Dockerize the service for production deployment.
- Integrate CI/CD for automated testing and deployment.

