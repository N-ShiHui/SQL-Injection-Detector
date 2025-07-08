import pandas as pd
import numpy as np
import pickle
import os
import hashlib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer # Convert features into numerical data

DATA_FILE = "sql.xlsx"
MODEL_FILE = "sql.pkl"
VECTORIZER_FILE = "tfidf_vector.pkl"
HASH_FILE = "data_hash.txt"

# === Hash helpers ===
def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def check_data_changes():
    current_hash = get_file_hash(DATA_FILE)
    if not os.path.exists(HASH_FILE):
        return True
    with open(HASH_FILE, "r") as f:
        saved_hash = f.read()
    return current_hash != saved_hash

def save_hash():
    with open(HASH_FILE, "w") as f:
        f.write(get_file_hash(DATA_FILE))

# === Train model ===
def train_and_evaluate_model():
    print("Loading and training model...")

    df = pd.read_excel(DATA_FILE)
    print(df.info())
    print(df.head(20))

    df['label'] = df['label'].astype(str)
    # === Split data ===
    X_train, X_temp, y_train, y_temp = train_test_split(df['query'], df['label'], test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.7, random_state=42)
    
    # === vectorize data ===
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    X_val = vectorizer.transform(X_val)

    # === train model ===
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(model)

    # === evaluate ===
    y_test_pred = model.predict(X_test)
    y_val_pred = model.predict(X_val)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    print(f'Test Accuracy: {accuracy_test}\nValidation Accuracy: {accuracy_val}')
    
    # === save model and vectorizer ===
    with open(MODEL_FILE, 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(VECTORIZER_FILE, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    
    save_hash()
    print("Model trained and saved.")

if __name__ == "__main__":
    if check_data_changes():
        train_and_evaluate_model()
    else:
        print("No change detected in dataset. Skipping retraining.")