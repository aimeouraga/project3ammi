import pytest

from tests import VALID_USERNAME, VALID_PASSWORD, client

def test_login_for_access_token():
    # Valid login test
    response = client.post("/token", data={"username": VALID_USERNAME, "password": VALID_PASSWORD})
    assert response.status_code == 200
    assert "access_token" in response.json()

    # Invalid login test
    response = client.post("/token", data={"username": "invalid_user", "password": "wrong_password"})
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()['detail']