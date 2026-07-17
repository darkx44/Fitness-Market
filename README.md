# Fitness Market Intelligence

Prédiction du nombre d'abonnements en salle de sport (`gym_memberships`) par pays et par année, à partir d'indicateurs socio-économiques et de santé, avec un modèle **XGBoost**.

Projet de machine learning de bout en bout : exploration des données, feature engineering, modélisation, interprétabilité (SHAP) et clustering de pays.

**🚀 Démo en ligne : [fitness-market.onrender.com/docs](https://fitness-market.onrender.com/docs)** — teste la prédiction directement depuis le navigateur (route `POST /predict`).

> Hébergée sur le tier gratuit de Render : la première requête après une période d'inactivité peut prendre ~30–60 s (réveil du service), puis les suivantes sont rapides.

---

## Problème

À partir de données par pays (démographie, richesse, santé, urbanisation), estimer le nombre d'abonnements en salle de sport. Utile pour dimensionner un marché ou repérer des zones à fort potentiel.

**Données** : 3 564 lignes, 14 colonnes (couples pays × année).
Cible : `gym_memberships` (entraînée en `log1p` pour stabiliser la distribution).

Principales variables : `country`, `year`, `region`, `number_of_gyms`, `urban_population_percentage`, `obesity_rate`, `gdp_per_capita_usd`, `population_total`, `average_membership_cost_usd`, `insufficient_physical_activity_pct`.

---

## Structure du projet

```
Projet Fitness_market/
├── data/                    # Données (non versionnées)
│   └── clean_gym_data.csv
├── notebooks/
│   ├── eda.ipynb            # Exploration des données
│   ├── feature_engineering.ipynb
│   ├── modeling.ipynb       # Entraînement et comparaison de modèles
│   ├── explainability.ipynb # Interprétabilité (SHAP)
│   └── country_clustering.ipynb  # Clustering (KMeans + PCA)
├── src/
│   ├── preprocessing.py     # Pipeline de preprocessing (ColumnTransformer)
│   ├── train.py             # Entraînement + sauvegarde des artefacts
│   └── predict.py           # Chargement du modèle et prédiction
├── models/                  # Modèle + préprocesseur sauvegardés (non versionnés)
├── requirements.txt
└── README.md
```

> `data/` et `models/` sont volontairement exclus du dépôt (voir `.gitignore`) pour ne pas versionner de fichiers lourds.

---

## Installation

```bash
# Créer et activer un environnement virtuel
python3 -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

Place le fichier `clean_gym_data.csv` dans le dossier `data/`.

---

## Utilisation

Les scripts se lancent **depuis la racine du projet** (pour que l'import `from src...` fonctionne) :

```bash
# Entraîner le modèle et sauvegarder les artefacts dans models/
python -m src.train

# Générer des prédictions avec le modèle sauvegardé
python -m src.predict
```

`train.py` affiche les métriques d'évaluation (R², MAE, RMSE) puis écrit `xgb_model.pkl` et `preprocessor.pkl` dans `models/`.

### Lancer l'API

Le modèle est exposé via une API **FastAPI** :

```bash
uvicorn app.main:app --reload
```

Puis :
- `http://127.0.0.1:8000/docs` — documentation interactive pour tester la route `POST /predict`
- `POST /predict` reçoit les 10 caractéristiques d'un pays (JSON) et renvoie le nombre d'abonnements prédit.

### Tests

```bash
pytest
```

---

## Méthode

- **Preprocessing** : encodage one-hot des variables catégorielles (`country`, `region`) via un `ColumnTransformer`, le reste passé tel quel.
- **Modèle** : `XGBRegressor` (200 arbres, `learning_rate=0.05`, `max_depth=5`, sous-échantillonnage 0.8).
- **Cible en log** : entraînement sur `log1p(gym_memberships)`, prédictions ramenées à l'échelle réelle avec `expm1`.
- **Interprétabilité** : analyse SHAP pour comprendre les variables qui pèsent le plus.
- **Clustering** : segmentation des pays par KMeans, visualisée avec une PCA.

---

## Pistes d'amélioration

- Exposer le modèle via une **API (FastAPI)** et une **démo Streamlit**.
- Ajouter des **tests** (`pytest`) sur le preprocessing et la prédiction.
- Empaqueter avec **Docker** et déployer.

---

## Stack

Python · pandas · scikit-learn · XGBoost · SHAP · matplotlib · seaborn
