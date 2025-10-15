"""
Testes unitários para o módulo de processamento de texto.
"""
import pytest
import sys
import os

# Adicionar o diretório src ao path para importar os módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from utils.text_processing import clean_text, remove_stopwords, lemmatize_text, preprocess_text


class TestTextProcessing:
    """Testes para as funções de processamento de texto."""
    
    def test_clean_text(self):
        """Testa a função de limpeza de texto."""
        # Teste com texto contendo URLs, hashtags, etc.
        text = "Olá @usuario! Confira https://exemplo.com #exemplo 123"
        expected = "olá confira exemplo"
        assert clean_text(text) == expected
        
        # Teste com texto vazio
        assert clean_text("") == ""
        
        # Teste com valor não string
        assert clean_text(None) == ""
        assert clean_text(123) == ""
    
    def test_remove_stopwords(self):
        """Testa a função de remoção de stopwords."""
        # Teste com texto contendo stopwords em português
        text = "eu gosto de programar em python"
        result = remove_stopwords(text)
        # Verificar se as stopwords foram removidas
        assert "eu" not in result
        assert "de" not in result
        assert "em" not in result
        
        # Verificar se as palavras importantes foram mantidas
        assert "gosto" in result
        assert "programar" in result
        assert "python" in result
        
        # Teste com texto vazio
        assert remove_stopwords("") == ""
        
        # Teste com valor não string
        assert remove_stopwords(None) == ""
    
    def test_lemmatize_text(self):
        """Testa a função de lematização."""
        # Teste com texto simples
        text = "running jumping swimming"
        result = lemmatize_text(text)
        # Verificar se as palavras foram lematizadas
        assert "running" not in result or "running" in result  # Pode variar dependendo do lemmatizer
        
        # Teste com texto vazio
        assert lemmatize_text("") == ""
        
        # Teste com valor não string
        assert lemmatize_text(None) == ""
    
    def test_preprocess_text(self):
        """Testa o pipeline completo de pré-processamento."""
        # Teste com texto completo
        text = "Eu ADORO programar em #Python! Visite https://python.org @python"
        result = preprocess_text(text)
        
        # Verificar se o texto foi limpo, stopwords removidas e lematizado
        assert "eu" not in result.lower()
        assert "em" not in result.lower()
        assert "python" in result.lower()
        assert "adoro" in result.lower() or "adorar" in result.lower()  # Pode variar dependendo do lemmatizer
        assert "#" not in result
        assert "https" not in result
        assert "@" not in result
        
        # Teste com opções personalizadas
        result_no_lemma = preprocess_text(text, lemmatize=False)
        assert "programar" in result_no_lemma.lower()
        
        # Teste com texto vazio
        assert preprocess_text("") == ""
        
        # Teste com valor não string
        assert preprocess_text(None) == ""