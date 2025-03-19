# Spam Detection Model  
![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.78.0-green) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24.2-orange) ![License](https://img.shields.io/badge/License-MIT-green)  

An advanced spam detection model using **Naive Bayes** and **TF-IDF vectorization** for accurate spam filtering. Built with **FastAPI** and deployed with **Uvicorn**.  

<p align="center">
    <img src="https://user-images.githubusercontent.com/674621/71187836-6f41f580-227a-11ea-9498-ffb7bb9aa4d5.gif" alt="Demo GIF" width="600"/>
</p>  

---

## 🌐 **Live Demo**  
👉 [**HiveMind Spam Detector**](https://hivemind-spam-detection-ml.onrender.com) – Try it live!  

---

## 🚀 **Features**  
✅ Classifies messages as **Spam** or **Not Spam**  
✅ High accuracy with TF-IDF + Multinomial Naive Bayes  
✅ Pre-trained model stored in `.pkl` format for fast loading  
✅ FastAPI-based REST API with minimal latency  

---

## 🏗️ **Project Structure**  
📂 HiveMind-Spam-Detection-ML
├── 📁 app/ # FastAPI app and routes
├── 📁 data/ # Training data (CSV)
├── 📁 models/ # Pre-trained ML models (.pkl)
├── 📄 requirements.txt # Dependencies
├── 📄 README.md # Project documentation

---

## 🛠️ **Setup**  
### 1️⃣ **Clone the repository**  

git https://github.com/Anshu-x/HiveMind-Spam-Detection-ML.git  
cd HiveMind-Spam-Detection-ML  
### 2️⃣ Create a virtual environment

python -m venv venv  
source venv/bin/activate    # For Linux/MacOS  
venv\Scripts\activate       # For Windows  
### 3️⃣ Install dependencies

pip install -r requirements.txt  
🚦 Run the App
Start the FastAPI server using Uvicorn:


uvicorn app.main:app --host 0.0.0.0 --port 8000
The app will be available at: http://localhost:8000/docs

## 🔍 Making Predictions
Send a POST request to the /predict endpoint:

Example using cURL:

curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "message": "You have won a free iPhone! Click the link to claim it."
}'
Example using Python Requests:

import requests

url = "http://localhost:8000/predict"
data = {"message": "You have won a free iPhone! Click the link to claim it."}
response = requests.post(url, json=data)
print(response.json())
✅ Sample Response

{
  "prediction": "Spam",
  "confidence": 0.92
}
🏋️ Training
Train the model using the dataset in the /data folder:

python app/train.py
👉 Adjust hyperparameters like alpha in train.py:

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=0.1)
👉 Save the model using joblib:

import joblib

joblib.dump(model, 'models/spam_detector.pkl')
## 💻 Code Overview
### ✅ Spam Classification Logic
Uses TF-IDF vectorization for feature extraction
Multinomial Naive Bayes for classification

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(data['message'])
y_train = data['label']

model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)

#Save model
joblib.dump(model, 'models/spam_detector.pkl')
### ✅ FastAPI Endpoint
/predict → POST request
Accepts message string
Returns classification and confidence score

from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load('models/spam_detector.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

class Message(BaseModel):
    message: str

@app.post('/predict')
def predict(msg: Message):
    input_vector = vectorizer.transform([msg.message])
    prediction = model.predict(input_vector)[0]
    confidence = model.predict_proba(input_vector).max()
    return {"prediction": prediction, "confidence": confidence}
## 📊 Performance
Metric	Value
Accuracy	96.5%
Precision	95.8%
Recall	94.6%
F1-Score	95.2%
## 🎯 Tech Stack
Tool/Library	Purpose
Python 3.10+	Core Programming Language
FastAPI	Web Framework
Scikit-learn	Machine Learning
Joblib	Model Serialization
Pandas	Data Handling
Uvicorn	ASGI Server
## 🚨 To-Do
 Improve classification for multilingual input
 Implement streaming classification
 Add logging and monitoring

## 💡 Contributions welcome! Feel free to submit a PR or open an issue. 👊

---
  
