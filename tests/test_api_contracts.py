from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_query_requires_question():
    response = client.post("/query", json={})

    assert response.status_code == 422


def test_debug_retrieve_requires_question():
    response = client.post("/debug/retrieve", json={})

    assert response.status_code == 422


def test_documents_endpoint_contract():
    response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()

    assert "documents" in data
    assert "count" in data
    assert "max_documents" in data