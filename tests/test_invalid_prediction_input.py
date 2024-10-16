import pytest

from tests import get_access_token, client

def test_invalid_prediction_input():
    token = get_access_token()
    headers = {"Authorization": f"Bearer {token}"}
    # Send an invalid prediction request (empty review)
    data = {"review": ""}  # Invalid input
    response = client.post("/predict", json=data, headers=headers)
    
    assert response.status_code == 422  # Expect validation error