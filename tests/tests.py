import pytest
from fastapi.testclient import TestClient
from app.app import   app
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





