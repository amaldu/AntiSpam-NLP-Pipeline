import pandas as pd
import re
import nlpaug.augmenter.word as naw
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Optional, Dict
from ..utils.textaug_techniques import TextAugmentation

stop_words = set(stopwords.words('english'))



def clean_types_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    
    df['Category'] = df['Category'].map({"ham": 0, "spam": 1}).astype(int)
    df['Message'] = df['Message'].astype(str)
    df.drop_duplicates(inplace=True)
    return df

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

# def preprocess(
#     df: pd.DataFrame, 
#     use_nlpaug: Optional[bool] = False, 
#     sa: float = None, 
#     aa: float = None) -> pd.DataFrame:

    
#     #NOTE - Spelling aug creates symbols and mayus so it has to come before cleaning
#     #NOTE - oversampling factors could be a list of ONE OR MORE items so the for has to be outside
#     #NOTE - spelling and synonim have lists and vals. the text augmentation function has to get one val
#     # per iteration so the forst have to be ouside too.
    
#     if use_nlpaug and sa is not None and aa is not None:
#         augmenter = TextAugmentation(sa=sa, aa=aa)
#         augmenter.load_config('experiments_config.yaml') 
#         augmented_data = augmenter.augment(df['Message'], df['Category']) 
        

        
#     df['Message'] = df['Message'].apply(clean_text)
    

    
#     df['Message'] = df['Message'].apply(word_tokenize)
#     df['Message'] = df['Message'].apply(remove_stopwords)
    
#     df['Message'] = df['Message'].apply(lemmatize_text)
    
#     return df
