from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

from trained_model import train_and_evaluate_model, check_data_changes # reuse training logic

# === Check model availability ===
if os.path.isfile("sql.pkl"):
    print("Model file found. Skipping retraining")
else:
    print("Model file not found. Training model now...")
    train_and_evaluate_model()

# === Initialize FastAPI ===
app = FastAPI(title="SQL Injection Detection API")

# === Load model and vectorizer at startup ===
def load_model_and_vectorizer(): 
    with open('sql.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vector.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# === Define input model for POST request ===
class QueryInput(BaseModel):
    query: str

# === Predict endpoint ===
@app.post("/detect")
def detect_sql_injection(input: QueryInput):
    query = input.query
    query_vector = vectorizer.transform([query])
    prediction = model.predict(query_vector)[0]
    result = "SQL Injection Detected" if prediction == '1' else "Safe"
    return {"query": query, "result": result}

# === Retrain endpoint ===
@app.post("/retrain")
def retrain_model():
    if check_data_changes():
        train_and_evaluate_model() # retrain and save model
        global model, vectorizer
        model, vectorizer = load_model_and_vectorizer() # reload new model
        return {"status": "Retrained due to new data."}
    else:
        return {"status": "No changes in data. Skipping retraining."}