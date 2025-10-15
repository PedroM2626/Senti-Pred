"""
Utilitários para processamento de texto no projeto Senti-Pred.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Garantir que os recursos do NLTK estejam disponíveis
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


def clean_text(text):
    """
    Limpa o texto removendo URLs, menções, hashtags, pontuação e números.
    
    Args:
        text (str): Texto a ser limpo
        
    Returns:
        str: Texto limpo
    """
    if not isinstance(text, str):
        return ""
    
    # Converter para minúsculas
    text = text.lower()
    # Remover URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remover menções e hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remover pontuação e caracteres especiais
    text = re.sub(r'[^\w\s]', '', text)
    # Remover números
    text = re.sub(r'\d+', '', text)
    # Remover espaços extras
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_stopwords(text, language='portuguese'):
    """
    Remove stopwords do texto.
    
    Args:
        text (str): Texto para remover stopwords
        language (str): Idioma das stopwords (padrão: 'portuguese')
        
    Returns:
        str: Texto sem stopwords
    """
    if not isinstance(text, str):
        return ""
    
    stop_words = set(stopwords.words(language))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    
    return ' '.join(filtered_text)


def lemmatize_text(text):
    """
    Aplica lematização ao texto.
    
    Args:
        text (str): Texto para lematizar
        
    Returns:
        str: Texto lematizado
    """
    if not isinstance(text, str):
        return ""
    
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokens]
    
    return ' '.join(lemmatized_text)


def preprocess_text(text, remove_stop=True, lemmatize=True, language='portuguese'):
    """
    Aplica todo o pipeline de pré-processamento ao texto.
    
    Args:
        text (str): Texto para processar
        remove_stop (bool): Se deve remover stopwords
        lemmatize (bool): Se deve aplicar lematização
        language (str): Idioma das stopwords
        
    Returns:
        str: Texto processado
    """
    if not isinstance(text, str):
        return ""
    
    # Limpar texto
    cleaned_text = clean_text(text)
    
    # Remover stopwords se solicitado
    if remove_stop:
        cleaned_text = remove_stopwords(cleaned_text, language)
    
    # Lematizar se solicitado
    if lemmatize:
        cleaned_text = lemmatize_text(cleaned_text)
    
    return cleaned_text