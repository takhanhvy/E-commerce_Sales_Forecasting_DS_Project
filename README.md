# Projet Data Science - Prévision des ventes e-commerce

Projet complet de Data Science basé sur le dataset Amazon Sale Report.
Objectif : prédire `Amount` (montant des ventes) avec un modèle supervisé (Ridge Regression) et produire un clustering non supervisé.

## Jeu de données
- **Nom** : Amazon Sale Report
- **Source** : Kaggle (Unlock Profits with E-Commerce Sales Data)
- **Fichier** : `Amazon Sale Report.csv`
- **Lien** : https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data

## Structure du projet
- `notebooks/eda.ipynb` notebook d'analyse exploratoire
- `src/` scripts de preprocessing et d'entraînement
- `api/` API Flask avec endpoints de prédiction
- `docs/` documents d'analyse métier et interprétation des métriques
- `data/raw/` données locales (CSV)
- `artifacts/` modèles et sorties
- `reports/` métriques et pipeline (créé après entraînement)
- `test/` tests unitaires

## Dataset (local)
Placer le fichier CSV ici :

`data/raw/Amazon Sale Report.csv`

## Installation
```bash
python3 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -r requirements.txt
```

## EDA
```bash
.venv/bin/python -m pip install jupyter
.venv/bin/jupyter notebook notebooks/eda.ipynb
```

## Entraînement supervisé
```bash
.venv/bin/python src/train_supervised.py
```

Sorties :
- `artifacts/regression_model.pkl`
- `reports/regression_metrics.json`
- `reports/pipeline_supervised.txt`

## Entraînement non supervisé (clustering)
```bash
.venv/bin/python src/train_unsupervised.py
```

Sorties :
- `artifacts/cluster_model.pkl`
- `reports/cluster_metrics.json`
- `reports/pipeline_cluster.txt`

## API (Flask)
Lancer l'API :
```bash
.venv/bin/python api/app.py
```

Tester une prédiction (supervisé) :
```bash
curl -X POST http://127.0.0.1:5000/predict_regression \
  -H "Content-Type: application/json" \
  -d '{"Date":"2023-01-01","Qty":2,"Category":"A","Sales Channel":"Online","Status":"Shipped"}'
```

Tester le clustering (non supervisé) :
```bash
curl -X POST http://127.0.0.1:5000/predict_cluster \
  -H "Content-Type: application/json" \
  -d '{"Date":"2023-01-01","Qty":2,"Category":"A","Sales Channel":"Online","Status":"Shipped"}'
```

## Tests
Exécuter les tests API :
```bash
python test/test_api.py
```

## Métriques et analyse business
- `docs/metrics_interpretation.md`
- `docs/business_analysis.md`
- `docs/model_justifications.md`

## Docker
Construire et lancer l'image :
```bash
docker build -t ecommerce-sales-api .
docker run -p 5000:5000 ecommerce-sales-api
```
