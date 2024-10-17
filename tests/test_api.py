import pytest
from data import clean_text
from model import load_model
from fastapi.testclient import TestClient
from app import  app
import os

# TestClient instance for FastAPI testing
client = TestClient(app)

# valid user credentials for testing
VALID_USERNAME = os.getenv("USER_NAME")
VALID_PASSWORD = os.getenv("PASSWORD")

# get  access token for authenticated requests
def get_access_token():
    response = client.post("/token", data={"username": VALID_USERNAME, "password": VALID_PASSWORD})
    return response.json()["access_token"]


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "API is running"



def test_load_model():
    model, device = load_model()
    assert device is not None, "Device should be set"
    assert model is not None, "Model should be loaded"



def test_preprocessing():
    sample_text = "This is a <b>great</b> movie!"
    cleaned_text = clean_text(sample_text).split()
    assert "great" in cleaned_text, "Stop words should be removed"
    assert "<b>" not in cleaned_text, "HTML tags should be removed"


def test_login_for_access_token():
    # Valid login test
    response = client.post("/token", data={"username": VALID_USERNAME, "password": VALID_PASSWORD})
    assert response.status_code == 200
    assert "access_token" in response.json()

    # Invalid login test
    response = client.post("/token", data={"username": "invalid_user", "password": "wrong_password"})
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()['detail']




