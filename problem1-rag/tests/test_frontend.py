import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_frontend_root_endpoint():
    """Verify GET / serves HTML index page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Cost-Efficient RAG Application" in response.text
    assert "chat-form" in response.text


def test_frontend_static_files():
    """Verify static CSS and JS entry points are served correctly."""
    css_res = client.get("/static/style.css")
    assert css_res.status_code == 200

    js_res = client.get("/static/app.js")
    assert js_res.status_code == 200
    assert "/api/v1/query" in js_res.text


def test_cors_headers_present():
    """Verify CORS headers are present on API requests."""
    response = client.options("/api/v1/query", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
    })
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] in ["*", "http://localhost:3000"]
