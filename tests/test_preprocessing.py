"""
Tests du preprocessing.

Pour les lancer, place-toi à la racine du projet et tape :  pytest
pytest trouve tout seul ce fichier (nom en test_*.py) et les fonctions test_*.
"""

import pandas as pd

from src.preprocessing import build_preprocessor


def make_fake_data():
    """
    Un petit jeu de données FACTICE qui imite la structure de tes vraies données.
    On n'a pas besoin du vrai CSV : quelques lignes suffisent pour tester la logique.
    """
    return pd.DataFrame(
        {
            "country": ["France", "Canada", "Japan"],
            "region": ["Europe", "Americas", "Asia"],
            "number_of_gyms": [4000, 6000, 5000],
            "gdp_per_capita_usd": [40000, 45000, 39000],
        }
    )


def test_preprocessor_garde_toutes_les_lignes():
    # Arrange : on prépare les données et l'outil à tester
    df = make_fake_data()
    preprocessor = build_preprocessor()

    # Act : on exécute le code qu'on veut vérifier
    result = preprocessor.fit_transform(df)

    # Assert : on affirme le résultat attendu
    # 3 lignes en entrée doivent donner 3 lignes en sortie (rien n'est perdu).
    assert result.shape[0] == 3


def test_preprocessor_encode_les_categories():
    df = make_fake_data()
    preprocessor = build_preprocessor()

    result = preprocessor.fit_transform(df)

    # Les colonnes texte (country, region) sont "éclatées" en plusieurs colonnes
    # par le one-hot encoding. On doit donc avoir PLUS de colonnes qu'au départ.
    assert result.shape[1] > df.shape[1]
