import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import os
import sys

# Add the parent directory to the path so we can import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app, is_tribal_request, build_analytics_prompt


class TestChatbotAPI:
    """Test suite for the chatbot API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test the root endpoint returns expected message"""
        response = client.get("/")
        assert response.status_code == 200
        assert "Chatbot de Política y Tribus API está funcionando" in response.json()["message"]
    
    @patch.dict(os.environ, {
        'GOOGLE_API_KEY': 'test_key',
        'INDEX_DIR': 'test_storage',
        'LLM_MODEL': 'models/gemini-1.5-flash',
        'EMBEDDING_MODEL': 'models/embedding-001'
    })
    def test_chat_endpoint_without_initialization(self, client):
        """Test chat endpoint returns 503 when chatbot is not initialized"""
        response = client.post("/chat", json={
            "query": "test query",
            "session_id": "test_session"
        })
        assert response.status_code == 503
        assert "no está inicializado" in response.json()["detail"]

    def test_health_endpoint_structure(self, client):
        """Test that health endpoint has expected structure"""
        response = client.get("/")
        assert response.status_code == 200
        assert isinstance(response.json(), dict)
        assert "message" in response.json()


class TestTribalDetection:
    """Test suite for tribal request detection functionality"""
    
    def test_is_tribal_request_positive_cases(self):
        """Test that tribal patterns are correctly detected"""
        tribal_queries = [
            "mándame el link de mi tribu",
            "envíame el enlace de referidos",
            "¿dónde está mi link de tribu?",
            "dame el enlace de mi grupo",
            "quiero entrar a mi tribu",
            "necesito mi link de referidos",
            "parce, mándame el link de mi tribu",
            "referido",
            "mis referidos"
        ]
        
        for query in tribal_queries:
            assert is_tribal_request(query), f"Failed to detect tribal request: {query}"
    
    def test_is_tribal_request_negative_cases(self):
        """Test that non-tribal queries are not detected as tribal"""
        non_tribal_queries = [
            "¿Cuáles son las propuestas de Daniel Quintero?",
            "¿Cómo puedo votar?",
            "¿Dónde está la sede de campaña?",
            "¿Qué opinas sobre la economía?",
            "Hola, ¿cómo estás?",
            "¿Cuándo son las elecciones?"
        ]
        
        for query in non_tribal_queries:
            assert not is_tribal_request(query), f"Incorrectly detected as tribal: {query}"
    
    def test_is_tribal_request_case_insensitive(self):
        """Test that tribal detection is case insensitive"""
        test_cases = [
            "MÁNDAME EL LINK DE MI TRIBU",
            "Referido",
            "MIS REFERIDOS",
            "Dame El Enlace De Mi Grupo"
        ]
        
        for query in test_cases:
            assert is_tribal_request(query), f"Case sensitivity failed for: {query}"


class TestAnalyticsPrompt:
    """Test suite for analytics prompt building functionality"""
    
    def test_build_analytics_prompt_basic(self):
        """Test analytics prompt building with basic data"""
        test_query = "¿Cómo está mi rendimiento?"
        test_analytics = {
            "name": "Test User",
            "ranking": {
                "today": {"position": 1, "points": 100},
                "week": {"position": 2, "points": 250},
                "month": {"position": 3, "points": 500}
            },
            "region": {"position": 5, "totalParticipants": 100},
            "city": {"position": 2, "totalParticipants": 20},
            "referrals": {
                "totalInvited": 10,
                "activeVolunteers": 5,
                "referralsThisMonth": 3,
                "conversionRate": 50,
                "referralPoints": 75
            }
        }
        
        prompt = build_analytics_prompt(test_query, test_analytics)
        
        # Verify that key elements are included in the prompt
        assert "Test User" in prompt
        assert "Posición #1" in prompt
        assert "100 puntos" in prompt
        assert "10 invitados" in prompt
        assert test_query in prompt
        assert "Bogotá" in prompt  # Should always reference Bogotá
        assert "Medellín" not in prompt  # Should never mention Medellín
    
    def test_build_analytics_prompt_empty_data(self):
        """Test analytics prompt building with empty data"""
        test_query = "¿Cómo estoy?"
        test_analytics = {}
        
        prompt = build_analytics_prompt(test_query, test_analytics)
        
        # Should handle empty data gracefully
        assert test_query in prompt
        assert "N/A" in prompt or "0" in prompt
    
    def test_build_analytics_prompt_partial_data(self):
        """Test analytics prompt building with partial data"""
        test_query = "¿Cuántos referidos tengo?"
        test_analytics = {
            "name": "Partial User",
            "referrals": {
                "totalInvited": 5
            }
        }
        
        prompt = build_analytics_prompt(test_query, test_analytics)
        
        assert "Partial User" in prompt
        assert "5 invitados" in prompt


class TestHealthChecks:
    """Test suite for health and monitoring endpoints"""


if __name__ == "__main__":
    pytest.main([__file__])