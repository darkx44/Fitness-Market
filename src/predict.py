import os
import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1. Configuration des chemins absolus (Pour éviter les ModuleNotFoundError)
# ------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "clean_gym_data.csv")

# ------------------------------------------------------------------
# 2. Chargement des artefacts (Modèle et Préprocesseur)
# ------------------------------------------------------------------
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# ------------------------------------------------------------------
# 3. Chargement des données à prédire
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Simulation de nouvelles données en retirant la cible (Y)
X_new = df.drop(
    columns=[
        "gym_memberships",
        "gym_memberships_log",
        "gym_penetration_rate",
        "fitness_participation_rate",
        "total_health_club_revenue_usd"
    ],
    errors="ignore"  # Sécurité si une colonne manque
)

# ------------------------------------------------------------------
# 4. Pipeline : Transformation et Prédiction
# ------------------------------------------------------------------
# Applique la pipeline de preprocessing
X_new_enc = preprocessor.transform(X_new)

# Prédiction (le modèle renvoie un log)
pred_log = model.predict(X_new_enc)

# Transformation inverse (car le modèle a été entraîné sur le log de la cible)
pred = np.expm1(pred_log)

# ------------------------------------------------------------------
# 5. Affichage du résultat
# ------------------------------------------------------------------
print("Top 10 des prédictions :")
print(pred[:10])

