import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_val_split(df: pd.DataFrame) -> tuple:
    
    X = df['Message']
    y = df['Category']  

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

        
