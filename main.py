from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API funzionante!"}

@app.post("/predict")
def predict(dati: dict):
    # Qui metterai la logica del tuo modello di machine learning
    # Per ora restituiamo un risultato di test
    return {"risultato": "test predizione", "dati_ricevuti": dati}

