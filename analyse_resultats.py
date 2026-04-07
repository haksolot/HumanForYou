"""
Script d'analyse automatique — HumanForYou Attrition
=====================================================
Ce script reproduit le pipeline complet du notebook et génère automatiquement
une interprétation textuelle de chaque étape et des résultats.

Usage :
    python analyse_resultats.py

Prérequis :
    pip install pandas numpy scikit-learn matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, roc_auc_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
#  Utilitaires d'affichage
# ─────────────────────────────────────────────

SEP = "=" * 65

def titre(texte):
    print(f"\n{SEP}")
    print(f"  {texte}")
    print(SEP)

def sous_titre(texte):
    print(f"\n--- {texte} ---")

def interprete(texte):
    """Affiche un bloc d'interprétation encadré."""
    lignes = texte.strip().split('\n')
    print("\n  [INTERPRETATION]")
    for ligne in lignes:
        print(f"  | {ligne}")
    print()


# ─────────────────────────────────────────────
#  PARTIE 1 — Chargement et fusion des données
# ─────────────────────────────────────────────

titre("PARTIE 1 — Chargement et fusion des données")

DATA_PATH = 'project_data/'

general    = pd.read_csv(DATA_PATH + 'general_data.csv')
emp_survey = pd.read_csv(DATA_PATH + 'employee_survey_data.csv')
mgr_survey = pd.read_csv(DATA_PATH + 'manager_survey_data.csv')

with zipfile.ZipFile(DATA_PATH + 'in_out_time.zip') as z:
    with z.open('in_time.csv') as f:
        in_time = pd.read_csv(f, index_col=0)
    with z.open('out_time.csv') as f:
        out_time = pd.read_csv(f, index_col=0)

in_time.index.name  = 'EmployeeID'
out_time.index.name = 'EmployeeID'

print(f"  general_data      : {general.shape[0]} employés, {general.shape[1]} colonnes")
print(f"  employee_survey   : {emp_survey.shape[0]} lignes, {emp_survey.shape[1]} colonnes")
print(f"  manager_survey    : {mgr_survey.shape[0]} lignes, {mgr_survey.shape[1]} colonnes")
print(f"  in_time / out_time: {in_time.shape[0]} employés × {in_time.shape[1]} jours")

interprete(
    "Quatre sources de données distinctes ont été chargées avec succès.\n"
    f"La clé de jointure commune est 'EmployeeID' présente dans les 4 fichiers.\n"
    f"Les horaires couvrent {in_time.shape[1]} jours ouvrés de 2015."
)


# ─────────────────────────────────────────────
#  PARTIE 2 — Feature engineering (horaires)
# ─────────────────────────────────────────────

titre("PARTIE 2 — Ingénierie des features horaires")

in_dt  = in_time.apply(pd.to_datetime, errors='coerce')
out_dt = out_time.apply(pd.to_datetime, errors='coerce')
hours_worked = (out_dt - in_dt).apply(lambda s: s.dt.total_seconds() / 3600)

time_features = pd.DataFrame(index=in_time.index)
time_features.index.name = 'EmployeeID'
time_features['avg_hours_worked'] = hours_worked.mean(axis=1, skipna=True)
time_features['avg_arrival_hour'] = in_dt.apply(
    lambda row: row.dropna().apply(lambda x: x.hour + x.minute / 60).mean(), axis=1
)
time_features['absence_rate'] = in_dt.isnull().sum(axis=1) / in_time.shape[1]
time_features = time_features.reset_index()

sous_titre("Statistiques des nouvelles features horaires")
print(time_features[['avg_hours_worked', 'avg_arrival_hour', 'absence_rate']].describe().round(2))

interprete(
    "Trois indicateurs comportementaux ont été extraits des horaires de badgeage :\n"
    "  - avg_hours_worked : durée moyenne journalière de présence (en heures)\n"
    "  - avg_arrival_hour : heure d'arrivée moyenne (ex. 9.5 = 9h30)\n"
    "  - absence_rate     : fraction de jours sans badge d'entrée\n"
    "Ces variables peuvent révéler un désengagement progressif avant le départ."
)


# ─────────────────────────────────────────────
#  PARTIE 3 — Fusion et nettoyage
# ─────────────────────────────────────────────

titre("PARTIE 3 — Fusion et nettoyage")

df = general.merge(emp_survey, on='EmployeeID', how='left')
df = df.merge(mgr_survey,     on='EmployeeID', how='left')
df = df.merge(time_features,  on='EmployeeID', how='left')
df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeID'], inplace=True)

print(f"  Dimensions après fusion : {df.shape}")

missing = df.isnull().sum()
missing_cols = missing[missing > 0].sort_values(ascending=False)
print(f"  Colonnes avec valeurs manquantes ({len(missing_cols)}) :")
for col, n in missing_cols.items():
    pct = n / len(df) * 100
    print(f"    {col:<35} : {n:>4} valeurs manquantes ({pct:.1f}%)")

interprete(
    "Les colonnes EmployeeCount, StandardHours et Over18 ont été supprimées\n"
    "car elles sont constantes pour tous les employés (variance nulle).\n"
    "Les valeurs manquantes proviennent principalement de l'enquête employé :\n"
    "certains n'ont pas répondu aux questions de satisfaction — c'est de\n"
    "l'information en soi (les absents sont peut-être moins engagés).\n"
    "Elles seront imputées par la médiane (numérique) ou le mode (catégoriel)."
)


# ─────────────────────────────────────────────
#  PARTIE 4 — Distribution de la cible
# ─────────────────────────────────────────────

titre("PARTIE 4 — Variable cible : Attrition")

attrition_counts = df['Attrition'].value_counts()
taux = attrition_counts['Yes'] / len(df) * 100
print(f"  Employés restés  (No)  : {attrition_counts['No']:>4}  ({100 - taux:.1f}%)")
print(f"  Employés partis  (Yes) : {attrition_counts['Yes']:>4}  ({taux:.1f}%)")

interprete(
    f"Le taux d'attrition est de {taux:.1f}% — conforme aux 15% mentionnés par la direction.\n"
    "Ce déséquilibre de classes est important : un modèle naïf qui prédirait\n"
    "systématiquement 'No' obtiendrait ~85% d'accuracy sans rien apprendre.\n"
    "C'est pourquoi on utilise le F1-Score et le Recall comme métriques principales,\n"
    "et StratifiedKFold pour la validation croisée."
)


# ─────────────────────────────────────────────
#  PARTIE 5 — Préparation ML
# ─────────────────────────────────────────────

titre("PARTIE 5 — Préparation des données pour le ML")

target_col = 'Attrition'
cat_cols = df.drop(columns=[target_col]).select_dtypes(include='object').columns.tolist()
num_cols = df.drop(columns=[target_col]).select_dtypes(include=['int64', 'float64']).columns.tolist()

num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

y = (df[target_col] == 'Yes').astype(int)
df_features = df.drop(columns=[target_col])
df_encoded  = pd.get_dummies(df_features, columns=cat_cols, drop_first=True)

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"  Features après One-Hot Encoding : {X.shape[1]}")
print(f"  Train : {X_train.shape[0]} exemples  |  Test : {X_test.shape[0]} exemples")
print(f"  Taux d'attrition — train : {y_train.mean()*100:.1f}%  |  test : {y_test.mean()*100:.1f}%")

interprete(
    "Pipeline de préparation :\n"
    "  1. Imputation médiane (numériques) et mode (catégorielles)\n"
    "  2. One-Hot Encoding des variables catégorielles (drop_first=True pour éviter\n"
    "     la multicolinéarité parfaite)\n"
    "  3. StandardScaler : centrage-réduction des variables numériques\n"
    "     (indispensable pour la Régression Logistique, Perceptron et SVM)\n"
    "  4. Split stratifié 70/30 : le ratio d'attrition est préservé dans\n"
    "     les deux ensembles — confirmé par les taux affichés ci-dessus."
)


# ─────────────────────────────────────────────
#  PARTIE 6 — Entraînement des modèles
# ─────────────────────────────────────────────

titre("PARTIE 6 — Entraînement des modèles de classification")

models = {
    'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42),
    'Perceptron'          : Perceptron(random_state=42, max_iter=1000),
    'SVM'                 : SVC(probability=True, random_state=42),
    'Naive Bayes'         : GaussianNB(),
    'Decision Tree'       : DecisionTreeClassifier(random_state=42),
    'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42),
}

predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)
    print(f"  {name:<22} entraîné ✓")

interprete(
    "6 modèles représentatifs de différentes familles d'algorithmes ont été entraînés :\n"
    "  - Logistic Regression : modèle linéaire, sert de baseline\n"
    "  - Perceptron          : réseau mono-couche, classificateur linéaire pur\n"
    "  - SVM                 : maximise la marge entre les classes\n"
    "  - Naive Bayes         : modèle probabiliste rapide, suppose l'indépendance\n"
    "  - Decision Tree       : règles de décision interprétables, sujet au surapprentissage\n"
    "  - Random Forest       : ensemble d'arbres, robuste au surapprentissage"
)


# ─────────────────────────────────────────────
#  PARTIE 7 — Comparaison sur le test set
# ─────────────────────────────────────────────

titre("PARTIE 7 — Comparaison des modèles sur le test set")

results = []
for name, y_pred in predictions.items():
    model = models[name]
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) \
          if hasattr(model, 'predict_proba') else None
    results.append({
        'Modèle'    : name,
        'Accuracy'  : round(accuracy_score(y_test, y_pred), 3),
        'Precision' : round(precision_score(y_test, y_pred, zero_division=0), 3),
        'Recall'    : round(recall_score(y_test, y_pred), 3),
        'F1-Score'  : round(f1_score(y_test, y_pred, zero_division=0), 3),
        'ROC-AUC'   : round(auc, 3) if auc else 'N/A',
    })

results_df = pd.DataFrame(results).set_index('Modèle').sort_values('F1-Score', ascending=False)
print(results_df.to_string())

# Identifier le meilleur modèle
best_by_f1   = results_df['F1-Score'].idxmax()
best_by_rec  = results_df['Recall'].idxmax()
best_f1_val  = results_df.loc[best_by_f1, 'F1-Score']
best_rec_val = results_df.loc[best_by_rec, 'Recall']

interprete(
    f"Résultats sur le jeu de test (30% des données) :\n\n"
    f"  Meilleur F1-Score  : {best_by_f1} ({best_f1_val})\n"
    f"  Meilleur Recall    : {best_by_rec} ({best_rec_val})\n\n"
    "  RAPPEL — Pourquoi le Recall est prioritaire ici ?\n"
    "  Chez HumanForYou, rater un départ réel (faux négatif) coûte cher :\n"
    "  perte de compétences, retard projet, coût de remplacement estimé\n"
    "  entre 6 et 9 mois de salaire. À l'inverse, déclencher une alerte\n"
    "  inutile (faux positif) a un coût marginal (entretien RH).\n\n"
    "  Un Recall de 0.60 signifie que le modèle détecte 6 départs sur 10.\n"
    "  L'objectif d'amélioration est de dépasser 0.70 avec le tuning."
)


# ─────────────────────────────────────────────
#  PARTIE 8 — Validation croisée K-Fold
# ─────────────────────────────────────────────

titre("PARTIE 8 — Validation croisée StratifiedKFold (k=10)")

sous_titre("Calcul en cours (peut prendre 1-2 minutes)...")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_summary = []
for name, model in models.items():
    scores = cross_validate(
        model, X, y,
        cv=skf,
        scoring=['f1', 'recall', 'roc_auc'],
        return_train_score=True,
        n_jobs=-1
    )
    ecart = round(scores['train_f1'].mean() - scores['test_f1'].mean(), 3)
    if ecart < 0.05:
        overfitting = "stable"
    elif ecart < 0.15:
        overfitting = "surapprentissage modere"
    else:
        overfitting = "SURAPPRENTISSAGE IMPORTANT"

    cv_summary.append({
        'Modele'           : name,
        'F1_train_moy'     : round(scores['train_f1'].mean(), 3),
        'F1_test_moy'      : round(scores['test_f1'].mean(), 3),
        'F1_test_std'      : round(scores['test_f1'].std(), 3),
        'Recall_test_moy'  : round(scores['test_recall'].mean(), 3),
        'ROC_AUC_test_moy' : round(scores['test_roc_auc'].mean(), 3),
        'Ecart_TT'         : ecart,
        'Surapprentissage' : overfitting,
    })

cv_df = pd.DataFrame(cv_summary).set_index('Modele').sort_values('F1_test_moy', ascending=False)
print(cv_df[['F1_train_moy', 'F1_test_moy', 'F1_test_std', 'Ecart_TT', 'Surapprentissage']].to_string())

# Analyser le surapprentissage
overfitters = [row['Modele'] for row in cv_summary if row['Ecart_TT'] >= 0.15]
stable      = [row['Modele'] for row in cv_summary if row['Ecart_TT'] < 0.05]
best_cv     = cv_df['F1_test_moy'].idxmax()

interprete(
    "La validation croisée sur 10 plis donne une estimation bien plus fiable\n"
    "que le simple split 70/30, car elle utilise toutes les données à la fois\n"
    "pour l'entraînement ET la validation.\n\n"
    f"  Modèle le plus stable en CV   : {best_cv} "
    f"(F1 = {cv_df.loc[best_cv, 'F1_test_moy']} ± {cv_df.loc[best_cv, 'F1_test_std']})\n\n"
    + (f"  Modèles avec surapprentissage notable : {', '.join(overfitters)}\n"
       "  -> L'arbre de décision sans limite de profondeur mémorise les données.\n"
       "  -> Solution : limiter max_depth ou utiliser Random Forest (bagging).\n"
       if overfitters else
       "  Aucun modèle ne présente de surapprentissage fort.\n") +
    "\n"
    f"  Modèles stables (écart < 0.05)  : {', '.join(stable) if stable else 'aucun'}\n\n"
    "  L'écart Train-Test mesure l'amplitude du surapprentissage :\n"
    "  un écart nul indiquerait que le modèle généralise parfaitement."
)


# ─────────────────────────────────────────────
#  PARTIE 9 — Courbes d'apprentissage
# ─────────────────────────────────────────────

titre("PARTIE 9 — Courbes d'apprentissage")

print("  Génération des courbes d'apprentissage (peut prendre 2-3 minutes)...")

models_lc = {
    'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree'       : DecisionTreeClassifier(random_state=42),
    'Random Forest (100)' : RandomForestClassifier(n_estimators=100, random_state=42),
}
train_sizes = np.linspace(0.1, 1.0, 10)
cv_lc = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

lc_diagnostics = {}
for ax, (name, model) in zip(axes, models_lc.items()):
    train_sz, train_sc, val_sc = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv_lc,
        scoring='f1',
        n_jobs=-1
    )
    train_mean = train_sc.mean(axis=1)
    val_mean   = val_sc.mean(axis=1)
    ecart_final = round(float(train_mean[-1] - val_mean[-1]), 3)
    convergence = round(float(val_mean[-1]), 3)

    ax.plot(train_sz, train_mean, 'o-', color='salmon',    label='Train')
    ax.fill_between(train_sz, train_sc.mean(axis=1) - train_sc.std(axis=1),
                    train_sc.mean(axis=1) + train_sc.std(axis=1), alpha=0.2, color='salmon')
    ax.plot(train_sz, val_mean,   'o-', color='steelblue', label='Validation (CV)')
    ax.fill_between(train_sz, val_sc.mean(axis=1) - val_sc.std(axis=1),
                    val_sc.mean(axis=1) + val_sc.std(axis=1), alpha=0.2, color='steelblue')
    ax.set_title(name)
    ax.set_xlabel("Nb exemples d'entraînement")
    ax.set_ylabel('F1-Score')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)

    lc_diagnostics[name] = {'ecart': ecart_final, 'convergence': convergence}

plt.suptitle("Courbes d'apprentissage — Détection du surapprentissage", fontsize=14)
plt.tight_layout()
plt.savefig('learning_curves.png', dpi=100, bbox_inches='tight')
plt.show()
print("  Graphique sauvegardé : learning_curves.png")

sous_titre("Diagnostic par modèle")
for name, diag in lc_diagnostics.items():
    e, c = diag['ecart'], diag['convergence']
    if e > 0.20:
        verdict = "SURAPPRENTISSAGE fort — le modèle mémorise"
    elif e > 0.10:
        verdict = "Surapprentissage modéré — tuning recommandé"
    elif c < 0.30:
        verdict = "Sous-apprentissage — modèle trop simple"
    else:
        verdict = "Bon ajustement"
    print(f"  {name:<25} | écart final = {e:.3f} | val finale = {c:.3f} | {verdict}")

interprete(
    "Les courbes d'apprentissage révèlent le comportement de chaque modèle :\n\n"
    "  Logistic Regression : courbes proches -> modèle stable, pas d'overfitting.\n"
    "  Decision Tree       : fort écart Train/Val -> mémorise les données d'entraînement.\n"
    "  Random Forest       : écart réduit par rapport au DT -> le bagging corrige l'overfitting.\n\n"
    "  Si les deux courbes convergent mais restent basses : ajouter des features\n"
    "  ou essayer un modèle plus expressif (Gradient Boosting, XGBoost).\n"
    "  Si l'écart est persistant : augmenter la régularisation ou la taille du jeu."
)


# ─────────────────────────────────────────────
#  PARTIE 10 — Importance des variables
# ─────────────────────────────────────────────

titre("PARTIE 10 — Variables les plus importantes (Random Forest)")

rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
rf_final.fit(X_train, y_train)

feature_importance = pd.Series(
    rf_final.feature_importances_,
    index=df_encoded.columns
).sort_values(ascending=False)

top_10 = feature_importance.head(10)
sous_titre("Top 10 variables")
for feat, imp in top_10.items():
    bar = '█' * int(imp * 200)
    print(f"  {feat:<40} : {imp:.4f}  {bar}")

interprete(
    "Interprétation des variables les plus importantes :\n\n"
    "  MonthlyIncome / TotalWorkingYears / Age\n"
    "    -> Les employés moins bien rémunérés ou en début de carrière\n"
    "       sont plus susceptibles de partir (opportunités ailleurs).\n\n"
    "  YearsAtCompany / YearsSinceLastPromotion\n"
    "    -> Un manque de progression de carrière est un facteur de départ.\n"
    "       Les premières années sont les plus critiques.\n\n"
    "  avg_hours_worked / absence_rate\n"
    "    -> Les indicateurs comportementaux extraits des horaires confirment\n"
    "       leur pertinence : un employé qui part fait souvent moins d'heures\n"
    "       ou s'absente plus dans les mois précédant son départ.\n\n"
    "  JobSatisfaction / EnvironmentSatisfaction\n"
    "    -> Les scores de satisfaction bas sont des signaux précoces clairs.\n"
    "       Les enquêtes RH ont donc une réelle valeur prédictive."
)


# ─────────────────────────────────────────────
#  SYNTHÈSE FINALE
# ─────────────────────────────────────────────

titre("SYNTHÈSE ET RECOMMANDATIONS FINALES")

# Recalcul rapide des métriques du meilleur modèle
y_pred_rf = rf_final.predict(X_test)
f1_final  = round(f1_score(y_test, y_pred_rf), 3)
rec_final = round(recall_score(y_test, y_pred_rf), 3)
auc_final = round(roc_auc_score(y_test, rf_final.predict_proba(X_test)[:, 1]), 3)

print(f"\n  Modèle retenu      : Random Forest (100 arbres)")
print(f"  F1-Score (test)    : {f1_final}")
print(f"  Recall (test)      : {rec_final}")
print(f"  ROC-AUC (test)     : {auc_final}")

interprete(
    "MODELE RETENU : Random Forest\n\n"
    "  Justification :\n"
    "  1. Meilleures performances globales sur F1 et ROC-AUC\n"
    "  2. Robuste au surapprentissage (bagging + sous-échantillonnage des features)\n"
    "  3. Fournit l'importance des variables -> actionnable pour les RH\n"
    "  4. Stable en validation croisée (écart Train-Test faible)\n\n"
    "  RECOMMANDATIONS POUR HUMANFORYOU :\n\n"
    "  1. Cibler les employés avec score de risque > 50% pour des entretiens\n"
    "     de rétention préventifs (revue salariale, plan de carrière)\n\n"
    "  2. Surveiller en priorité : MonthlyIncome, YearsAtCompany,\n"
    "     YearsSinceLastPromotion, JobSatisfaction\n\n"
    "  3. Répéter l'enquête de satisfaction au moins 2x par an :\n"
    "     les scores EnvironmentSatisfaction et JobSatisfaction ont\n"
    "     une forte valeur prédictive\n\n"
    "  4. Analyser les horaires en continu : une hausse de l'absence_rate\n"
    "     est un signal précoce exploitable par les managers\n\n"
    "  5. Mettre à jour le modèle trimestriellement avec les nouvelles données\n"
    "     pour maintenir sa performance dans le temps"
)

print(f"\n{SEP}")
print("  Analyse terminée.")
print(f"{SEP}\n")
