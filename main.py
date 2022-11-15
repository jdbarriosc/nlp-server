from fastapi import FastAPI
from DataModel import DataModel
from joblib import load

app = FastAPI()

@app.post("/is-suicidal")
def make_predictions(data: DataModel ):
    tfidf = load(open("assets/vectorizer.joblib", "rb"))
    imported_model = load("assets/SVM_suicidio_proy1.joblib")
    vector = tfidf.transform([data.text])
    prediction = imported_model.predict(vector)[0]
    result = int(prediction)
    return result

