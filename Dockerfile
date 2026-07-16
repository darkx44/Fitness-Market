# ------------------------------------------------------------------
# Dockerfile = la recette pour construire l'image de ton API.
# Chaque ligne est une étape ; Docker les exécute de haut en bas.
# ------------------------------------------------------------------

# 1. On part d'une image de base officielle : un mini-Linux avec Python 3.12
#    déjà installé. "slim" = version allégée (image plus petite).
FROM python:3.12-slim

# 2. Dossier de travail à l'intérieur du conteneur. Toutes les commandes
#    suivantes s'exécutent depuis /code.
WORKDIR /code

# 3. On copie D'ABORD la liste des dépendances, puis on l'installe.
#    Pourquoi séparer ? Docker met en cache cette étape : tant que
#    requirements-api.txt ne change pas, il ne réinstalle pas tout à
#    chaque modification de code → builds beaucoup plus rapides.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 4. On copie le reste du projet (code, modèle...) dans le conteneur.
COPY . .

# 5. On documente le port sur lequel l'API écoute.
EXPOSE 8000

# 6. La commande lancée au démarrage du conteneur.
#    --host 0.0.0.0 est OBLIGATOIRE dans Docker : sans ça, l'API n'écoute
#    que "l'intérieur" du conteneur et tu ne peux pas y accéder depuis ton Mac.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
