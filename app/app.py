from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from model.model import load_model, predict_sentiment
from data.preprocessing import preprocess_text
from auth import Token, authenticate_user, User, get_current_active_user, create_access_token
from datetime import timedelta, datetime, timezone
from dotenv import load_dotenv
from typing import Union, Optional
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

# Startup event: Load the model and tokenizer during app startup
@app.on_event("startup")
def startup_event():
    global model, device
    model, device = load_model()

# Function to save predictions to a CSV on Azure Blob Storage
def save_prediction_to_csv(data: dict):
    try:
        blob_client = container_client.get_blob_client(csv_blob_name)
        download_stream = blob_client.download_blob()
        existing_data = pd.read_csv(io.BytesIO(download_stream.readall()))
    except Exception:
        existing_data = pd.DataFrame(columns=["username", "review", "sentiment", "confidence", "request_time", "response_time"])

    new_data = pd.DataFrame([data])
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)

    with io.BytesIO() as output_stream:
        updated_data.to_csv(output_stream, index=False)
        output_stream.seek(0)
        blob_client.upload_blob(output_stream, overwrite=True)

# Function to combine reviews from the CSV file into a single text
def extract_text_from_csv(file: UploadFile) -> str:
    df = pd.read_csv(file.file)
    text = " ".join(df['review'].astype(str).tolist())
    return text

# Define the prediction endpoint that handles both text and CSV
@app.post("/predict")
async def predict(text: Optional[str] = Form(None), file: Optional[UploadFile] = File(None), current_user: User = Depends(get_current_active_user)):
    input_text = None

    try: 
        # Ensure that only one of text or file is provided, not both
        if text and file:
            return {"error": "Please provide either a review or a CSV file, not both."}

        # If single text input (review) is provided
        if text:
            input_text = text
            input_tensor = preprocess_text(input_text)
            sentiment_label, confidence_score = predict_sentiment(model, device, input_tensor)

            return {
                "review": input_text,
                "sentiment": sentiment_label,
                "Model Confidence Score": confidence_score,
            }
        
        # If a file is uploaded (CSV file case)
        if file:
            if not file.filename.endswith(".csv"):
                return {"error": "Unsupported file type. Please upload a CSV file."}

            # Combine all reviews from the CSV file into a single text
            input_text = extract_text_from_csv(file)

            input_tensor = preprocess_text(input_text)
            sentiment_label, confidence_score = predict_sentiment(model, device, input_tensor)

            return {
                "combined_reviews": input_text[:500],  # Show first 500 characters of combined text for debugging
                "overall_sentiment": sentiment_label,
                "Model Confidence Score": confidence_score,
            }

        # If neither review text nor file is provided
        return {"error": "Please provide either a review or upload a CSV file for sentiment analysis."}

    except Exception as e:
        # Log the error to investigate any issues
        logger.error(f"Error: {e}")
        return {"error": str(e), "message": "An error occurred during the prediction process."}

# Token generation for authentication
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
