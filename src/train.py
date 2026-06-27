import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

from src.preprocessing import build_preprocessor


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


df = pd.read_csv('/Users/bigkems/Desktop/Projets Perso/Formation AI-engineer/Projet Fitness_market/data/clean_gym_data.csv')

df["gym_memberships_log"] = np.log1p(df["gym_memberships"])

# definition de la target
y = df["gym_memberships_log"]

# definition des Features
X = df.drop(columns=["gym_memberships",
    "gym_memberships_log",
    "gym_penetration_rate",
    "fitness_participation_rate",
    "total_health_club_revenue_usd"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


preprocessor = build_preprocessor()

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Creation du model
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Entrainemeng du model
xgb_model.fit(X_train_encoded, y_train)

# Prediction du model
y_pred_xgb = xgb_model.predict(X_test_encoded)


import os
import joblib

# Définir le chemin de sauvegarde de manière propre (comme dans predict.py)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Sauvegarder le modèle et le préprocesseur (remplace par le nom de tes variables)
joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_model.pkl"))
joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))

print("Modèles sauvegardés avec succès dans le dossier models/ !")



print("TRAIN FINISHED")
