from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert "collection_name" in data
    assert "doc_chunks" in data
    assert "documents" in data
    assert "max_documents" in data