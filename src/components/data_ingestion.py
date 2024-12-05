import os,sys
from dataclasses import dataclass


import pandas as pd 
import re
from string import punctuation
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

import logging

INPUT_FILE = "data/input_data.csv"  # Ruta al archivo CSV
DB_FILE = "data/ingestion.db"       # Ruta al archivo de SQLite
TABLE_NAME = "ingested_data"        # Nombre de la tabla

def read_data(file_path):
    logging.info(f"Reading the data from {file_path}")
    try:
        train = pd.read_csv(file_path)
        test = pd.read_csv(file_path)
        val = pd.read_csv(file_path)
        return train, test, val
    except Exception as e:
        logging.error(f"Error reading data: {e}")
        raise

def process_data(data):
    logging.info(f"Procesando datos...")
    try:



        logging.info(f"Procesamiento finalizado.")
        return data
    except Exception as e:
        logging.error(f"Error al procesar datos: {e}")
        raise

def save_to_database(data, db_file, table_name):
    """Guarda datos en una base de datos SQLite."""
    info(f"Guardando datos en {db_file}, tabla {table_name}...")
    try:
        with sqlite3.connect(db_file) as conn:
            data.to_sql(table_name, conn, if_exists="replace", index=False)
        info(f"Datos guardados correctamente.")
    except Exception as e:
        error(f"Error al guardar datos: {e}")
        raise

df = pd.read_csv("../data/bronze/spam.csv")




df['Category'] = df['Category'].map({"ham": 0, "spam": 1}).astype(int)
df['Message'] = df['Message'].astype(str)



df_duplicated = df['Message'].duplicated().sum()




df.drop_duplicates(inplace=True)




def clean_text(text):
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



df['Message']=df['Message'].apply(clean_text)
df['Message'] = df['Message'].apply(word_tokenize)
df['Message'] = df['Message'].apply(lambda x: [word for word in x if word not in stop_words])

lemmatizer=WordNetLemmatizer()
def lem_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

df['Message'] = df['Message'].apply(lem_tokens)
df = df[df['Message'].str.strip().astype(bool)]





df.to_csv("../data/silver/df_preprocessed.csv", index= False)

