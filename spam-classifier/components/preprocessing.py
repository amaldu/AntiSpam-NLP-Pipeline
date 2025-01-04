import pandas as pd
import re
import nlpaug.augmenter.word as naw
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Optional

stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    special_replacements = {
        r"£": "pound",
        r"\$": "dollar",
        r"\€": "euro",
        r"%": "percentage", 
        r"ì": "i",
        r"ü": "you",
    }
    
    emoticon_pattern = re.compile(r"""
    [:;=Xx]           
    [-~]?             
    [\)\]\(\[dDpP/]   
    """, re.VERBOSE)
    
    for pattern, replacement in special_replacements.items():
        text = re.sub(pattern, replacement, text)
    
    text = re.sub(emoticon_pattern, 'emoji', text)
    text = text.lower()
    
    text = re.sub('<[^<>]+>', ' ', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub('[0-9]+', 'number', text)
    text = re.sub('[^\s]+@[^\s]+', 'emailaddr', text)
    
    text = text.translate(str.maketrans('', '', punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(tokens: list) -> str:
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def lemmatize_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

def preprocess(df: pd.DataFrame, use_nlpaug: Optional[bool] = False) -> pd.DataFrame:
    """
    Function to preprocess the data.
    - Converts 'ham'/'spam' labels to 0/1 in 'Category'
    - Cleans text in 'Message' column using `clean_text`
    - Optionally applies text augmentation with `nlpaug`
    - Tokenizes and removes stopwords
    - Lemmatizes the text

    Args:
    df (pd.DataFrame): Input DataFrame with 'Category' and 'Message' columns.
    use_nlpaug (bool): Whether to apply NLP augmentation before lemmatizing.

    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    
    df['Category'] = df['Category'].map({"ham": 0, "spam": 1}).astype(int)
    df['Message'] = df['Message'].astype(str)
    
    df.drop_duplicates(inplace=True)
    
    df['Message'] = df['Message'].apply(clean_text)
    
    # Si se especifica, aplicamos NLP augmentation
    if use_nlpaug:
        aug = naw.SynonymAug(aug_p=0.1)  # Ejemplo de augumentación de sinónimos
        df['Message'] = df['Message'].apply(aug.augment)
    
    # Tokenización y eliminación de stopwords
    df['Message'] = df['Message'].apply(word_tokenize)
    df['Message'] = df['Message'].apply(remove_stopwords)
    
    # Lematización del texto
    df['Message'] = df['Message'].apply(lemmatize_text)
    
    return df
