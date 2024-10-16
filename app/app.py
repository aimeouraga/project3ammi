# from fastapi import FastAPI, Depends, HTTPException, status
# from fastapi.security import OAuth2PasswordRequestForm
# from pydantic import BaseModel
# from model.model import load_model, predict_sentiment
# from data.preprocessing import preprocess_text
# from auth.auth import Token, authenticate_user, User, get_current_active_user, create_access_token
# from datetime import timedelta
# # from dotenv import load_dotenv
# import os
# from dotenv import load_dotenv
# import os
# load_dotenv()

# # Load environment variables
# # load_dotenv()

# # Initialize FastAPI
# app = FastAPI()

# # Global variables for model and device
# model = None
# device = None

# ACCESS_TOKEN_EXPIRE_MINUTES = 30

# # Define the input data structure using Pydantic
# class ReviewInput(BaseModel):
#     review: str

# # Startup event: Load the model and tokenizer during app startup
# @app.on_event("startup")
# def startup_event():
#     global model, device
#     model, device = load_model()

# # Define the prediction endpoint
# @app.post("/predict")
# def predict(data: ReviewInput, current_user: User = Depends(get_current_active_user)):
#     try:
#         # Preprocess the review text
#         input_tensor = preprocess_text(data.review)
        
#         # Make the prediction
#         sentiment_label, probability = predict_sentiment(model, device, input_tensor)
        
#         # Return the result
#         return {"review": data.review, "sentiment": sentiment_label, "Model Confidence Score": probability}
    
#     except Exception as e:
#         return {"error": str(e), "message": "An error occurred during the prediction process."}

# @app.post("/token", response_model=Token)
# async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
#     user = authenticate_user(form_data.username, form_data.password)
#     if not user:
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
#                             detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
#     access_token_expires = timedelta(minutes=int(ACCESS_TOKEN_EXPIRE_MINUTES))
#     access_token = create_access_token(
#         data={"sub": user.username}, expires_delta=access_token_expires)
#     return {"access_token": access_token, "token_type": "bearer"}

# @app.get("/users/me/", response_model=User)
# async def read_users_me(current_user: User = Depends(get_current_active_user)):
#     return current_user

# @app.get("/health")
# def health_check():
#     return {"status": "API is running", "model_loaded": model is not None}

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from model.model import load_model, predict_sentiment
from data.preprocessing import preprocess_text
from auth.auth import Token, authenticate_user, User, get_current_active_user, create_access_token
from datetime import timedelta, datetime, timezone
from dotenv import load_dotenv
import os
import io
import pandas as pd
import time
import logging
from azure.storage.blob import BlobServiceClient
from opencensus.ext.azure.log_exporter import AzureLogHandler

load_dotenv()

# Azure Blob Storage setup
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_name = "project2"
container_client = blob_service_client.get_container_client(container_name)
csv_blob_name = "second_project_predict_log.csv"

# Application Insights logger setup
APP_INSIGHTS_KEY = os.getenv("AZURE_APP_INSIGHTS_INSTRUMENTATION_KEY")
logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(connection_string=f'InstrumentationKey={APP_INSIGHTS_KEY}'))
logger.setLevel(logging.INFO)

app = FastAPI()

# Global variables for model and device
model = None
device = None

ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Define the input data structure using Pydantic
class ReviewInput(BaseModel):
    review: str

# Startup event: Load the model and tokenizer during app startup
@app.on_event("startup")
def startup_event():
    global model, device
    model, device = load_model()

# Function to save predictions to a CSV on Azure Blob Storage
def save_prediction_to_csv(data: dict):
    # Load the existing CSV from Azure Blob Storage
    try:
        blob_client = container_client.get_blob_client(csv_blob_name)
        download_stream = blob_client.download_blob()
        existing_data = pd.read_csv(io.BytesIO(download_stream.readall()))
    except Exception:
        # If CSV does not exist, create an empty DataFrame
        existing_data = pd.DataFrame(columns=["username", "review", "sentiment", "confidence", "request_time", "response_time"])

    # Add the new prediction data
    new_data = pd.DataFrame([data])
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)

    # Save the updated CSV back to Azure Blob Storage
    with io.BytesIO() as output_stream:
        updated_data.to_csv(output_stream, index=False)
        output_stream.seek(0)
        blob_client.upload_blob(output_stream, overwrite=True)

# prediction endpoint
@app.post("/predict")
def predict(data: ReviewInput, current_user: User = Depends(get_current_active_user)):
    try:
        # Record request time
        request_time = datetime.now(timezone.utc)
        
        # Preprocess the review text
        input_tensor = preprocess_text(data.review)

        
        start_time = time.time()

        # Make the prediction
        sentiment_label, probability = predict_sentiment(model, device, input_tensor)

        response_time = time.time() - start_time  
        
        # Create a log entry for Application Insights
        logger.info(f"User: {current_user.username} - Prediction: {sentiment_label} - Confidence: {probability}")
        
        # Log the prediction to Azure Blob Storage as CSV
        save_prediction_to_csv({
            "username": current_user.username,
            "review": data.review,
            "sentiment": sentiment_label,
            "confidence": probability,
            "request_time": request_time,
            "response_time": response_time
        })

        # Return the result
        return {"review": data.review, "sentiment": sentiment_label, "Model Confidence Score": probability, "Response Time (s)": response_time}

    except Exception as e:
        return {"error": str(e), "message": "An error occurred during the prediction process."}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=int(ACCESS_TOKEN_EXPIRE_MINUTES))
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.get("/health")
def health_check():
    return {"status": "API is running", "model_loaded": model is not None}
