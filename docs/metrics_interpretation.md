# Interprétation des métriques

## Objectif
Ce document explique les métriques de performance produites après entraînement.
Le projet contient :

- un **modèle supervisé de régression** pour prédire `Amount` (montant des ventes par commande, en INR)
- un **modèle non supervisé de clustering** pour segmenter les commandes en groupes homogènes

## Modèle supervisé (régression) : métriques et valeurs

Les résultats sont comparés à une baseline `DummyRegressor(strategy="mean")` (prédit la moyenne), afin de vérifier que le modèle apprend bien des relations utiles.

### Comparaison Baseline vs Ridge Regression

| Métrique | Baseline (Dummy) | Modèle (Ridge Regression) |
| :--- | ---: | ---: |
| **MAE** | 225.26 | 169.07 |
| **RMSE** | 290.25 | 229.61 |
| **R²** | -0.0039 | 0.3718 |

Données d'évaluation : `rows_train = 96,944` et `rows_test = 24,236` (split temporel si la colonne date est présente).

### Comment interpréter chaque métrique

1. **MAE (Mean Absolute Error)**
   - **Définition** : écart moyen en valeur absolue entre la prédiction et la valeur réelle.
   - **Unité** : même unité que `Amount` (INR).
   - **Interprétation** : avec une MAE de **169.07 INR**, le modèle se trompe en moyenne d'environ **169 INR** par commande.
   - **Comparaison** : vs baseline (**225.26 INR**), l'erreur moyenne baisse d'environ **25%** (gain ≈ 56 INR/commande).

2. **RMSE (Root Mean Squared Error)**
   - **Définition** : racine de l'erreur quadratique moyenne ; pénalise davantage les grosses erreurs.
   - **Unité** : INR.
   - **Interprétation** : RMSE **229.61 INR** indique que des écarts importants existent encore (commandes atypiques / montants extrêmes).
   - **Comparaison** : vs baseline (**290.25 INR**), baisse d'environ **21%**.
   - **Lecture MAE vs RMSE** : l'écart (RMSE > MAE) suggère la présence d'outliers ; le modèle gère mieux la majorité des cas que les cas extrêmes.

3. **R² (coefficient de détermination)**
   - **Définition** : proportion de variance expliquée par le modèle (1 = parfait, 0 = équivalent à la moyenne, négatif = pire que la moyenne).
   - **Interprétation** : **R² = 0.3718** signifie que le modèle explique environ **37%** de la variabilité de `Amount` sur le jeu de test.
   - **Comparaison** : la baseline est légèrement négative (**-0.0039**), ce qui confirme que prédire la moyenne ne capture pas la structure des données, alors que Ridge apporte un gain substantiel.

### Conclusion (supervisé)

Le modèle **Ridge Regression** surpasse clairement la baseline sur toutes les métriques.

- La baisse de MAE/RMSE indique une amélioration tangible en INR (utile pour la planification).
- Le R² (~37%) montre un pouvoir explicatif réel, mais aussi une marge d'amélioration : une partie importante des variations de `Amount` provient probablement de facteurs non présents dans les features (promotions, retours, concurrence, événements, etc.).

## Modèle non supervisé (clustering) : métriques et interprétation

Le clustering vise à **segmenter** des commandes similaires pour aider l'analyse marketing/logistique (profilage, ciblage, optimisation de process).

### Valeurs obtenues (MiniBatchKMeans)

- **Meilleur k** (sélection silhouette sur échantillon) : **2**
- **Silhouette score** (échantillon de 5,000 lignes) : **0.1310**
- **Distribution des clusters** (échantillon) :
  - Cluster 0 : 2712 (~54%)
  - Cluster 1 : 2288 (~46%)

### Comment interpréter

- **Silhouette score** : proche de 1 = clusters bien séparés ; proche de 0 = clusters qui se chevauchent.
  - Ici (**0.131**) : séparation **faible à modérée**. Le clustering capte probablement quelques différences globales, mais la segmentation n'est pas très "tranchée".
- **Distribution** : les clusters sont relativement équilibrés, ce qui est positif (pas de cluster ultra-dominant), mais il faut compléter par une analyse qualitative (profils moyens par cluster : `Qty`, `Category`, `Sales Channel`, `Status`, etc.).

### Conclusion (non supervisé)

Le clustering produit une segmentation exploitable pour explorer des profils, mais le score silhouette suggère que :

- soit les variables actuelles ne suffisent pas à créer des segments fortement séparés,
- soit le comportement client est naturellement continu (pas de frontières nettes).

Une étape recommandée est de décrire les clusters (top catégories, taille, canaux, statuts) pour vérifier l'intérêt business de cette segmentation.
