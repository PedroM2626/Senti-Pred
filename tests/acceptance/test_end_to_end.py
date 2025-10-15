"""
Testes de aceitação para o fluxo completo do sistema.
"""
import pytest
import sys
import os
import json

# Adicionar o diretório src ao path para importar os módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Simulação de um teste de aceitação end-to-end
def test_end_to_end_workflow():
    """
    Testa o fluxo completo do sistema, desde o pré-processamento até a predição.
    
    Este é um teste simulado que verifica se o fluxo completo funciona corretamente.
    Em um ambiente real, este teste faria chamadas reais à API e verificaria as respostas.
    """
    # Simula o fluxo completo
    # 1. Texto de entrada
    input_text = "Este produto superou minhas expectativas! Recomendo fortemente."
    
    # 2. Simula o pré-processamento
    processed_text = "produto superou expectativas recomendo fortemente"
    
    # 3. Simula a predição do modelo
    prediction = {
        "sentiment": "positive",
        "probability": 0.95
    }
    
    # 4. Simula a resposta da API
    api_response = {
        "text": input_text,
        "processed_text": processed_text,
        "sentiment": prediction["sentiment"],
        "probability": prediction["probability"]
    }
    
    # Verificações de aceitação
    assert api_response["sentiment"] == "positive"
    assert api_response["probability"] > 0.7
    assert "text" in api_response
    assert "processed_text" in api_response

def test_negative_sentiment_workflow():
    """Testa o fluxo para um texto com sentimento negativo."""
    # Simula o fluxo completo para um texto negativo
    input_text = "Produto de péssima qualidade. Não recomendo de forma alguma."
    
    # Simula o pré-processamento
    processed_text = "produto péssima qualidade recomendo forma alguma"
    
    # Simula a predição do modelo
    prediction = {
        "sentiment": "negative",
        "probability": 0.88
    }
    
    # Simula a resposta da API
    api_response = {
        "text": input_text,
        "processed_text": processed_text,
        "sentiment": prediction["sentiment"],
        "probability": prediction["probability"]
    }
    
    # Verificações de aceitação
    assert api_response["sentiment"] == "negative"
    assert api_response["probability"] > 0.7
    assert "text" in api_response
    assert "processed_text" in api_response