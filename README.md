# Using Random Forest Classifier to Detect SQL Injection Attack
### This project focuses on the use of a machine learning technique, Random Forest Classifier, to detect SQL injection attack.
### Data used is a clean input based off SQL queries.
### Steps to run the model and fastapi app:
### 1) Run trained_model.py file to train model and generate 'sql.pkl', 'tfidf_vector.pkl' and 'data_hash.txt' files.
### 2) Run FastAPI app in main.py by typing in "uvicorn main:app --reload" in terminal 
### 3) In the website url, type in http://127.0.0.1:8000/doc
### 4) POST a query to /detect to check a SQL query for injection attacks OR 
### 5) Retrain the model via /retrain if there are changes to the dataset.
