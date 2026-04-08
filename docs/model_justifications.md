# Justification des modèles et métriques

## Modèle supervisé

### Algorithme retenu
- **Ridge Regression** (régression linéaire régularisée) est choisie pour gérer un grand nombre de variables après encodage catégoriel, tout en limitant le sur-apprentissage.
- Ce modèle est stable, rapide à entraîner et compatible avec une matrice sparse issue du One-Hot Encoding.

### Métriques retenues
- **MAE** : mesure l'erreur moyenne en valeur absolue, interprétable directement en unité de la cible.
- **RMSE** : pénalise davantage les grosses erreurs, utile pour détecter les prédictions très éloignées.
- **R2** : indique la proportion de variance expliquée par le modèle.

## Modèle non supervisé

### Algorithme retenu
- **MiniBatchKMeans** est choisi pour sa rapidité et sa capacité à traiter des volumes importants.
- Il permet de segmenter les commandes en groupes homogènes, utile pour des analyses marketing ou logistique.

### Métriques retenues
- **Silhouette score** : mesure la séparation entre clusters, plus élevé = meilleure segmentation.
- **Distribution des clusters** : vérifie l'équilibre et l'utilité des segments (évite un cluster dominant).
