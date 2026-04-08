# Analyse business

## Impact métier

L'objectif principal du projet est de transformer les données de commandes en un outil d'aide à la décision. En pratique, la prévision de `Amount` (montant des ventes par commande) et la segmentation (clustering) permettent de passer d'une gestion réactive à une gestion plus proactive.

### Cas d'usage (régression sur `Amount`)

- **Pilotage des ventes et planification**
  - Estimer le niveau de ventes attendu (par période, par catégorie, par canal) pour ajuster les objectifs, les ressources et les priorités.
  - Anticiper les périodes de forte demande et préparer les équipes (SAV, préparation, expédition).

- **Optimisation des stocks et du réapprovisionnement**
  - Identifier les catégories/tailles/statuts de commande associés aux montants les plus élevés.
  - Ajuster les politiques de stock (seuils de réappro, safety stock) pour réduire les ruptures et limiter le sur-stock.

- **Marketing et allocation de budget**
  - Adapter le calendrier des campagnes (et la pression marketing) aux périodes où la demande est attendue plus forte.
  - Prioriser les segments (catégories / canaux) où l'élasticité semble plus favorable, plutôt que de répartir le budget uniformément.

- **Finance et cash-flow**
  - La prévision des montants attendus aide à mieux anticiper les encaissements et à réduire l'incertitude sur le chiffre d'affaires.
  - Elle peut servir de base à des scénarios (pessimiste / central / optimiste) pour le pilotage.

### Cas d'usage (clustering)

Le clustering vise à regrouper les commandes en profils homogènes (ex: commandes à faible montant vs plus élevé, ou profils liés au canal/statut).

- **Segmentation opérationnelle**
  - Adapter les workflows (préparation/expédition) selon le type de commande.
  - Détecter des comportements atypiques (profils rares ou incohérents) à investiguer.

- **Segmentation marketing**
  - Comprendre des profils de commandes (catégories, quantités, canaux) pour affiner ciblage et messages.
  - Aider à construire des recommandations simples (ex: pousser une catégorie sur un canal donné).

## Limites

- **Cible = chiffre d'affaires brut, pas la rentabilité**
  - `Amount` mesure un montant de vente, mais ne reflète pas les coûts (expédition, retours, remises, coûts d'acquisition). Le modèle optimise donc la prédiction du CA, pas directement le profit.

- **Variables exogènes manquantes**
  - Certaines variations de ventes peuvent être dues à des facteurs non inclus : promotions, événements, concurrence, disponibilité produit, ruptures, changements de prix.

- **Qualité et stabilité des données**
  - Valeurs manquantes, incohérences de saisie, et valeurs extrêmes peuvent dégrader la performance.
  - Risque de dérive dans le temps (concept drift) si les habitudes d'achat ou l'offre produit changent.

- **Limites liées au modèle (Ridge Regression)**
  - Modèle linéaire : performant et robuste, mais peut manquer des relations non linéaires (interactions complexes entre catégorie, canal, quantité, saisonnalité).

- **Limites de la segmentation (clustering)**
  - Le score silhouette observé est faible, ce qui suggère que les profils ne sont pas fortement séparés : la segmentation doit être validée qualitativement (profil moyen de chaque cluster) avant usage opérationnel.

## Perspectives

- **Enrichissement des features**
  - Ajouter des variables calendrier (jour de la semaine, semaine, mois, vacances/jours fériés si applicable).
  - Ajouter des interactions simples (ex: catégorie x canal) et des indicateurs de saisonnalité.

- **Objectif business plus proche de la valeur**
  - Étendre l'approche à une prédiction de **marge** (ou profit net) en intégrant les coûts logistiques, retours et remises.
  - Passer d'un objectif "montant par commande" à des KPI actionnables : ventes par catégorie/semaine, volumes à expédier, taux de retour.

- **Modèles et évaluation**
  - Tester des modèles non linéaires performants sur tabulaire (Gradient Boosting, XGBoost/LightGBM si autorisé) pour potentiellement améliorer MAE/RMSE.
  - Mettre en place une validation temporelle plus systématique (backtesting) si l'usage visé est la prévision dans le temps.

- **Mise en production et gouvernance**
  - Définir un monitoring (distribution des features, dérive, performance au fil du temps) et une stratégie de ré-entraînement.
  - Documenter les seuils de décision : à partir de quelle baisse/hausse prévisionnelle on déclenche un réapprovisionnement ou une campagne.

## Conclusion

Les résultats montrent que le modèle supervisé fait nettement mieux qu'une baseline naïve et fournit une base exploitable pour l'aide à la décision. Le projet a un impact direct sur la capacité à anticiper la demande et à prioriser les actions (stock, marketing, pilotage).

Pour maximiser la valeur métier, les prochaines étapes consistent à : (1) rapprocher la cible des enjeux de rentabilité (marge), (2) enrichir les données avec des variables externes, et (3) valider l'usage opérationnel via un suivi dans le temps et une analyse qualitative des segments issus du clustering.
