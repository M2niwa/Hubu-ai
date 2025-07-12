import os
import pytest
from fastapi.testclient import TestClient
from api.main import app
from config.settings import settings

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    # 在测试环境中强制设置 API_KEY
    monkeypatch.setenv("API_KEY", "testkey")
    # 重新加载 settings
    from importlib import reload
    import config.settings
    reload(config.settings)

def test_missing_api_key():
    r = client.post("/ask", json={"question": "Hello", "use_local": True})
    assert r.status_code == 403
    assert "Invalid or missing API Key" in r.json()["detail"]

def test_invalid_api_key():
    r = client.post(
        "/ask",
        headers={"X-API-KEY": "badkey"},
        json={"question": "Hello", "use_local": True}
    )
    assert r.status_code == 403

def test_bad_request_payload():
    # valid API key but missing body -> 422 Unprocessable Entity
    r = client.post("/ask", headers={"X-API-KEY": "testkey"})
    assert r.status_code == 422
