"""
API Fitness Market — point d'entrée.

Pour la lancer, depuis la racine du projet :
    uvicorn app.main:app --reload

Puis ouvre dans ton navigateur :
    http://127.0.0.1:8000        -> message de vie de l'API
    http://127.0.0.1:8000/docs   -> documentation interactive (pour tester /predict)
"""

import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ------------------------------------------------------------------
# 1. Chargement du modèle et du préprocesseur — UNE SEULE FOIS au démarrage.
#    (On ne les recharge pas à chaque requête : ce serait lent et inutile.)
# ------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model = joblib.load(os.path.join(BASE_DIR, "models", "xgb_model.pkl"))
preprocessor = joblib.load(os.path.join(BASE_DIR, "models", "preprocessor.pkl"))

app = FastAPI(title="Fitness Market API")


# ------------------------------------------------------------------
# 2. Le SCHÉMA d'entrée (Pydantic).
#    Chaque champ = une caractéristique attendue, avec son type.
#    IMPORTANT : l'ordre des champs suit EXACTEMENT l'ordre des colonnes
#    vues à l'entraînement — c'est ce que le préprocesseur attend.
# ------------------------------------------------------------------
class CountryFeatures(BaseModel):
    country: str
    year: int
    region: str
    number_of_gyms: float
    urban_population_percentage: float
    obesity_rate: float
    gdp_per_capita_usd: float
    population_total: float
    average_membership_cost_usd: float
    insufficient_physical_activity_pct: float


# ------------------------------------------------------------------
# 3. Les routes
# ------------------------------------------------------------------
@app.get("/")
def racine():
    return {"message": "API Fitness Market en ligne"}


# POST : on ENVOIE des données, l'API les traite et renvoie une prédiction.
@app.post("/predict")
def predict(features: CountryFeatures):
    # Pydantic a déjà validé 'features'. On le transforme en DataFrame
    # d'une seule ligne (le format qu'attend le préprocesseur).
    df = pd.DataFrame([features.model_dump()])

    # Même chaîne qu'à l'entraînement : preprocessing -> prédiction.
    X = preprocessor.transform(df)
    pred_log = model.predict(X)

    # Le modèle a appris sur le log de la cible : on repasse à l'échelle réelle.
    prediction = float(np.expm1(pred_log)[0])

    return {"gym_memberships_pred": round(prediction)}
