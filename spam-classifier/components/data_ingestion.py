import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
import os 

import yaml

def read_data(config_file: str) -> pd.DataFrame:
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    base_path = config['dataset']['path']
    dataset_file = config['dataset']['file']
    
    dataset_path = os.path.join(base_path, dataset_file)
    
    logging.info(f"Reading the dataset from {dataset_path}")
    
    try:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset '{dataset_path}' does not exist.")
        
        data = pd.read_csv(dataset_path)

        if data.empty:
            raise ValueError(f"The dataset '{dataset_path}' is empty or malformed.")
        
        return data

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"The dataset '{dataset_path}' is empty.")
        raise ValueError(f"The dataset '{dataset_path}' is empty and cannot be processed.")
    except pd.errors.ParserError:
        logging.error(f"Error parsing the dataset '{dataset_path}'.")
        raise ValueError(f"There was an error parsing the dataset '{dataset_path}'.")
    except Exception as e:
        logging.error(f"Error reading the data: {e}")
        raise ValueError(f"An error occurred while reading the dataset '{dataset_path}': {e}")





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

