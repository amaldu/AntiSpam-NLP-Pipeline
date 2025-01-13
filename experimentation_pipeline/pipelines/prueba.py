from dotenv import load_dotenv
import os

load_dotenv()  # Cargar el archivo .env

print(os.getenv("PYTHONPATH"))  # Verifica que la variable esté cargada correctamente

cd.. 

print(os.getcwd())
