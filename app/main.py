from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# Initialize FastAPI app
app = FastAPI()

# Load Model & Vectorizer from disk
model_path = os.path.join(os.getcwd(), 'models', 'model.pkl')
vectorizer_path = os.path.join(os.getcwd(), 'models', 'vectorizer.pkl')

# Load the model and vectorizer
vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)

# Define the request body structure using Pydantic
class TextInput(BaseModel):
    text: str

# Home route
@app.get("/")
def home():
    return {"message": "Welcome to HiveMind Spam Detection API!"}

# Predict route to classify input text as spam or ham
@app.post("/predict/")
def predict_spam(input_data: TextInput):
    """Predict if the given text is spam or ham"""
    text = input_data.text
    
    # Transform the input text using the vectorizer
    text_transformed = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_transformed)[0]
    
    # Return prediction result
    return {"text": text, "prediction": "spam" if prediction == 1 else "ham"}
