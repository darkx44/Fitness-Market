import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

from src.preprocessing import build_preprocessor


# ------------------------------------------------------------------
# Chemins relatifs au projet (fonctionne sur n'importe quelle machine)
# ------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "clean_gym_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

df = pd.read_csv(DATA_PATH)

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

# ------------------------------------------------------------------
# Évaluation (sur l'échelle d'origine, pas le log)
# ------------------------------------------------------------------
y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(y_pred_xgb)

r2 = r2_score(y_test, y_pred_xgb)
mae = mean_absolute_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))

print(f"R2 (log)   : {r2:.4f}")
print(f"MAE        : {mae:,.0f}")
print(f"RMSE       : {rmse:,.0f}")

# ------------------------------------------------------------------
# Sauvegarde des artefacts
# ------------------------------------------------------------------
import joblib

os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_model.pkl"))
joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))

print("Modeles sauvegardes avec succes dans le dossier models/ !")
print("TRAIN FINISHED")
