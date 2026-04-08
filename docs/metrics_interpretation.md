# Interprétation des métriques

## Objectif
Ce document sert à expliquer les métriques du modèle supervisé (régression sur Amount).

## Métriques utilisées
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R2 (coefficient de détermination)
- Baseline (DummyRegressor)

## Comment interpréter
- MAE : erreur moyenne en valeur absolue. Plus bas = mieux.
- RMSE : pénalise davantage les grosses erreurs. Plus bas = mieux.
- R2 : proche de 1 = bon pouvoir explicatif. Peut être négatif si le modèle est mauvais.
- Baseline : référence minimale. Le modèle doit faire mieux que la baseline.

## Commentaires à remplir après entraînement
- Valeurs obtenues (MAE, RMSE, R2)
- Comparaison avec la baseline
- Explication des écarts
- Justification du choix de l'algorithme
