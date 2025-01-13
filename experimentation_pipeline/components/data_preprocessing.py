import pandas as pd
import re
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import numpy as np
from ..utils.textaug_techniques import TextAugmentation

stop_words = set(stopwords.words('english'))



def clean_types_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Starting data cleaning: mapping 'Category' and converting 'Message' to string...")
        df['email'] = df['email'].map({"ham": 0, "spam": 1}).astype(int)
        logging.info("'email' column successfully converted to integer.")
        
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        final_count = len(df)
        logging.info(f"Duplicates removed: {initial_count - final_count} rows dropped.")
        
        return df
    
    except KeyError as e:
        logging.error(f"KeyError: Missing expected column in DataFrame - {e}")
        raise
    
    except ValueError as e:
        logging.error(f"ValueError: Issue with data conversion - {e}")
        raise
    
    except Exception as e:
        logging.error(f"Unexpected error during data cleaning: {e}")
        raise



def clean_text(text: str) -> str:
    special_replacements = {
        r"£": "pound",
        r"$": "dollar",
        r"€": "euro",
        r"%": "percentage",
        r"♣": "clover", 
        r"®": "registered trademark",
        r"©": "copyright",
        r"☺": "emoji",
        r"™": "trademark",
    }
    
    chat_words = {
        "afaik": "As Far As I Know",
        "afk": "Away From Keyboard",
        "asap": "As Soon As Possible",
        "atk": "At The Keyboard",
        "atm": "At The Moment",
        "a3": "Anytime, Anywhere, Anyplace",
        "bak": "Back At Keyboard",
        "bbl": "Be Back Later",
        "bbs": "Be Back Soon",
        "bfn": "Bye For Now",
        "b4n": "Bye For Now",
        "brb": "Be Right Back",
        "brt": "Be Right There",
        "btw": "By The Way",
        "b4": "Before",
        "b4n": "Bye For Now",
        "cu": "See You",
        "cul8r": "See You Later",
        "cya": "See You",
        "faq": "Frequently Asked Questions",
        "fc": "Fingers Crossed",
        "fwiw": "For What It's Worth",
        "fyi": "For Your Information",
        "gal": "Get A Life",
        "gg": "Good Game",
        "gn": "Good Night",
        "gmta": "Great Minds Think Alike",
        "gr8": "Great!",
        "g9": "Genius",
        "ic": "I See",
        "icq": "I Seek you (also a chat program)",
        "ilu": "ILU: I Love You",
        "imho": "In My Honest/Humble Opinion",
        "imo": "In My Opinion",
        "iow": "In Other Words",
        "irl": "In Real Life",
        "kiss": "Keep It Simple, Stupid",
        "ldr": "Long Distance Relationship",
        "lmao": "Laugh My A.. Off",
        "lol": "Laughing Out Loud",
        "ltns": "Long Time No See",
        "l8r": "Later",
        "mte": "My Thoughts Exactly",
        "m8": "Mate",
        "nrn": "No Reply Necessary",
        "oic": "Oh I See",
        "pita": "Pain In The A..",
        "prt": "Party",
        "prw": "Parents Are Watching",
        "qpsa?": "Que Pasa?",
        "rofl": "Rolling On The Floor Laughing",
        "roflol": "Rolling On The Floor Laughing Out Loud",
        "rotflmao": "Rolling On The Floor Laughing My A.. Off",
        "sk8": "Skate",
        "stats": "Your sex and age",
        "asl": "Age, Sex, Location",
        "thx": "Thank You",
        "ttfn": "Ta-Ta For Now!",
        "ttyl": "Talk To You Later",
        "u": "You",
        "u2": "You Too",
        "u4e": "Yours For Ever",
        "wb": "Welcome Back",
        "wtf": "What The F...",
        "wtg": "Way To Go!",
        "wuf": "Where Are You From?",
        "w8": "Wait...",
        "7k": "Sick:-D Laugher",
        "tfw": "That feeling when",
        "mfw": "My face when",
        "mrw": "My reaction when",
        "ifyp": "I feel your pain",
        "tntl": "Trying not to laugh",
        "jk": "Just kidding",
        "idc": "I don't care",
        "ily": "I love you",
        "imu": "I miss you",
        "adih": "Another day in hell",
        "zzz": "Sleeping, bored, tired",
        "wywh": "Wish you were here",
        "time": "Tears in my eyes",
        "bae": "Before anyone else",
        "fimh": "Forever in my heart",
        "bsaaw": "Big smile and a wink",
        "bwl": "Bursting with laughter",
        "bff": "Best friends forever",
        "csl": "Can't stop laughing"
    }

    
    emoticon_pattern = re.compile(r"""
    [:;=Xx]           
    [-~]?             
    [\)\]\(\[dDpP/]   
    """, re.VERBOSE)
    text = text.lower()
    
    
    tokens = [re.sub(pattern, replacement, token) for token in tokens for pattern, replacement in special_replacements.items()]
    tokens = [token.replace('\n', ' ') for token in tokens]
    tokens = [re.sub(emoticon_pattern, 'emoji', token) for token in tokens]
    tokens = [token.lower() for token in tokens]
    tokens = [re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, token) for token in tokens for abbr, full_form in chat_words.items()]
    tokens = [re.sub('<[^<>]+>', ' ', token) for token in tokens]
    tokens = [re.sub(r'http\S+|www.\S+', '', token) for token in tokens]
    tokens = [re.sub(r'[0-9]+', 'number', token) for token in tokens]
    tokens = [re.sub(r'[^\s]+@[^\s]+', 'emailaddr', token) for token in tokens]
    tokens = [token.translate(str.maketrans('', '', punctuation)) for token in tokens]
    tokens = [re.sub(r'[^a-zA-Z\s]', '', token) for token in tokens]
    tokens = [re.sub(r'\s+', ' ', token).strip() for token in tokens]
    return tokens



def remove_stopwords(tokens: list) -> list:
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def lemmatize_text(tokens: list) -> str:
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

def preprocess(X_train: pd.Series, X_val: pd.Series, X_test: pd.Series, 
               y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> tuple:
    try:
        logging.info("Starting text preprocessing...")
        
        processed_texts = []

        for text in X_train:
            logging.debug("Cleaning text...")
            cleaned_text = clean_text(str(text))
            
            logging.debug("Tokenizing text...")
            tokens = word_tokenize(cleaned_text)
            
            logging.debug("Removing stopwords...")
            text_without_stopwords = remove_stopwords(tokens)
            
            logging.debug("Lemmatizing text...")
            preprocessed_text = lemmatize_text(text_without_stopwords)
            
            processed_texts.append(preprocessed_text)

        X_train_processed = np.array(processed_texts)
        
        X_train = X_train_processed
        X_val = X_val.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_val = y_val.to_numpy()
        y_test = y_test.to_numpy()

        logging.info("Text preprocessing completed successfully.")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise ValueError(f"An error occurred during preprocessing: {e}")