# HumanForYou — Prédiction de l'Attrition des Employés

Projet d'Intelligence Artificielle réalisé dans le cadre du bloc IA à CESI.

## Contexte

L'entreprise pharmaceutique **HumanForYou** (Inde, ~4 000 employés) subit un taux de rotation annuel de **15 %**. Ce projet vise à identifier les facteurs d'influence et à construire un modèle de classification capable de prédire quels employés risquent de quitter l'entreprise, afin de permettre des actions de rétention ciblées.

## Structure du projet

```
HumanForYou/
├── HumanForYou_Attrition.ipynb   # Notebook principal (pipeline ML complet)
├── analyse_resultats.py          # Script autonome avec interprétation automatique
├── project_data/
│   ├── general_data.csv          # Données RH démographiques (cible : Attrition)
│   ├── employee_survey_data.csv  # Enquête qualité de vie au travail (juin 2015)
│   ├── manager_survey_data.csv   # Évaluations managers (février 2015)
│   └── in_out_time.zip           # Horaires de badgeage journaliers (2015)
└── README.md
```

## Plan du notebook

| Partie | Contenu |
|--------|---------|
| 1 | Préparation de l'environnement |
| 2 | Chargement et fusion des 4 sources de données |
| 3 | Analyse exploratoire (EDA) : distributions, corrélations, boxplots |
| 4 | Ingénierie des features horaires (`avg_hours_worked`, `absence_rate`, `avg_arrival_hour`) |
| 5 | Préparation ML : imputation, One-Hot Encoding, StandardScaler, split 70/30 |
| 6 | Entraînement de 6 modèles de classification |
| 7 | Comparaison des modèles : métriques, matrices de confusion, courbes ROC |
| 7.5 | Validation croisée StratifiedKFold (k=10) + détection du surapprentissage |
| 7.6 | Courbes d'apprentissage (Learning Curves) |
| 8 | Optimisation du meilleur modèle via GridSearchCV |
| 9 | Importance des variables et recommandations |

## Modèles utilisés

- Régression Logistique
- Perceptron
- SVM (Support Vector Machine)
- Naive Bayes
- Arbre de Décision
- **Random Forest** *(modèle retenu)*

## Métriques prioritaires

Le **Recall** est la métrique principale : manquer un départ réel (faux négatif) coûte bien plus cher (remplacement, formation, retard projet) que déclencher une alerte inutile (faux positif).

## Utilisation

### Notebook Jupyter

```bash
jupyter notebook HumanForYou_Attrition.ipynb
```

### Script d'analyse avec interprétation automatique

```bash
python analyse_resultats.py
```

Le script exécute le pipeline complet et génère une interprétation textuelle automatique à chaque étape (chargement, EDA, modèles, CV, learning curves, recommandations).

## Dépendances

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

| Bibliothèque | Version recommandée | Usage |
|---|---|---|
| pandas | >= 1.3 | Manipulation des données |
| numpy | >= 1.21 | Calcul numérique |
| scikit-learn | >= 1.0 | Modèles ML, métriques, pipelines |
| matplotlib | >= 3.4 | Visualisations |
| seaborn | >= 0.11 | Visualisations statistiques |

## Sources des données

Données générées issues du projet Kaggle : [HR Analytics Case Study](https://www.kaggle.com/vjchoudhary7/hr-analytics-case-study), anonymisées par le service RH de HumanForYou.
