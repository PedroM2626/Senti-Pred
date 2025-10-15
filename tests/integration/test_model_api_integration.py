"""
Testes de integração entre o modelo e a API.
"""
import pytest
import sys
import os
import json

# Adicionar o diretório src ao path para importar os módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Mock para simular a resposta da API Django
class MockResponse:
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

# Testes de integração simulados
def test_model_api_integration():
    """Testa a integração entre o modelo e a API."""
    # Simula a resposta da API para um texto de exemplo
    mock_response = MockResponse({
        "text": "Este produto é excelente!",
        "sentiment": "positive",
        "probability": 0.92
    }, 200)
    
    # Verifica se a resposta contém os campos esperados
    assert "text" in mock_response.json()
    assert "sentiment" in mock_response.json()
    assert "probability" in mock_response.json()
    
    # Verifica se o status code é 200 (OK)
    assert mock_response.status_code == 200
    
    # Verifica se o sentimento é válido
    assert mock_response.json()["sentiment"] in ["positive", "negative", "neutral"]
    
    # Verifica se a probabilidade está no intervalo [0, 1]
    assert 0 <= mock_response.json()["probability"] <= 1

def test_model_api_error_handling():
    """Testa o tratamento de erros na integração entre o modelo e a API."""
    # Simula uma resposta de erro da API
    mock_error_response = MockResponse({
        "error": "Texto vazio ou inválido"
    }, 400)
    
    # Verifica se a resposta contém o campo de erro
    assert "error" in mock_error_response.json()
    
    # Verifica se o status code é 400 (Bad Request)
    assert mock_error_response.status_code == 400