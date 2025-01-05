import pandas as pd
import re
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from ..utils.textaug_techniques import TextAugmentation

stop_words = set(stopwords.words('english'))



def clean_types_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    
    df['Category'] = df['Category'].map({"ham": 0, "spam": 1}).astype(int)
    df['Message'] = df['Message'].astype(str)
    df.drop_duplicates(inplace=True)
    return df


################
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
    return filtered_tokens

def lemmatize_text(tokens: list) -> str:
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)


def preprocess(text: str) -> str:
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    text_without_stopwords = remove_stopwords(tokens)
    preprocessed_text = lemmatize_text(text_without_stopwords)
    return preprocessed_text
