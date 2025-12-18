import sys
import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(root_dir)

from models.pytorch.torch_classification import Classification
from data.utils.data_loader import load_data

# =================CONFIGURATION=================
MODEL_PATH = "models/pytorch/saved_models/heart_disease.pth"
THRESHOLD = 0.3532
# ===============================================

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Chargement de l'environnement...")
df = load_data()
target_col = "Heart_Disease_Yes"

X_all = df.drop(columns=target_col).values
scaler = StandardScaler()
scaler.fit(X_all)

random_idx = np.random.randint(0, len(df))
patient_row = df.iloc[random_idx]

# On sépare les Features (X) de la Réponse réelle (y)
patient_features_raw = patient_row.drop(target_col).values.reshape(1, -1)
real_status = int(patient_row[target_col])

# 4. Préparation pour le modèle (Normalisation)
# On utilise le scaler calibré plus haut
patient_features_scaled = scaler.transform(patient_features_raw)
patient_tensor = torch.tensor(patient_features_scaled, dtype=torch.float32).to(device)

# 5. Chargement du Modèle
input_dim = X_all.shape[1]
# IMPORTANT : Il faut la EXACTE même architecture que le train
model = Classification(input_dim, hidden_layers=[256, 128, 64]).to(device)

# On charge les poids sauvegardés
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # Mode évaluation (désactive le Dropout)
except FileNotFoundError:
    print(f"ERREUR : Le fichier {MODEL_PATH} est introuvable.")
    print("As-tu bien ajouté la ligne torch.save() à la fin de ton entraînement ?")
    sys.exit()

# 6. Prédiction
with torch.no_grad():
    logit = model(patient_tensor)
    prob = torch.sigmoid(logit).item()  # Probabilité entre 0 et 1

# Décision basée sur ton seuil calibré
prediction = 1 if prob > THRESHOLD else 0

# =================AFFICHAGE=================
print("\n" + "=" * 40)
print(f"     FICHE PATIENT #{random_idx}")
print("=" * 40)

# Affichage des infos disponibles (On utilise .get() pour éviter le crash si la colonne manque)
print(f"IMC (BMI)            : {patient_row.get('BMI', 'N/A'):.1f}")

# Gestion de la colonne Fumeur (Nom corrigé : Smoking_History_Yes)
is_smoker = patient_row.get("Smoking_History_Yes", 0)
print(f"Historique Fumeur    : {'Oui' if is_smoker == 1 else 'Non'}")

# Gestion de l'alcool (Nom corrigé : Alcohol_Consumption)
algo_cons = patient_row.get("Alcohol_Consumption", 0)
print(
    f"Conso. Alcool        : {algo_cons} unités/mois"
)  # ou par semaine selon ton dataset

# Gestion de l'exercice (Exercise_Yes)
is_sporty = patient_row.get("Exercise_Yes", 0)
print(f"Activité Physique    : {'Oui' if is_sporty == 1 else 'Non'}")

# Gestion du Sexe (Sex_Male)
is_male = patient_row.get("Sex_Male", 0)
print(f"Sexe                 : {'Homme' if is_male == 1 else 'Femme'}")

# Gestion de l'âge (Complexe car encodé en plusieurs colonnes Age_Category_XX)
# On cherche la colonne d'âge qui est à 1 (True)
age_cols = [c for c in df.columns if "Age_Category_" in c]
current_age = "Inconnu"
for col in age_cols:
    if patient_row[col] == 1:
        current_age = col.replace("Age_Category_", "")
        break
print(f"Tranche d'âge        : {current_age}")


print("-" * 40)
print(f"Probabilité calculée : {prob * 100:.2f}%")
# Rappel : On affiche le seuil calibré utilisé
print(f"Seuil de décision    : {THRESHOLD}")
print("-" * 40)

# Résultat visuel
if prediction == 1:
    print("🚨 PRÉDICTION DU MODÈLE : À RISQUE (Positif)")
else:
    print("✅ PRÉDICTION DU MODÈLE : SAIN (Négatif)")

print("-" * 40)
if real_status == 1:
    print("REALITÉ MÉDICALE      : MALADE")
else:
    print("REALITÉ MÉDICALE      : SAIN")

# Vérification finale
if prediction == real_status:
    print("\nResultat : CORRECT 👍")
else:
    if prediction == 1 and real_status == 0:
        print("\nResultat : FAUX POSITIF (Fausse alerte, mais prudent) ⚠️")
    else:
        print("\nResultat : FAUX NÉGATIF (Malade raté) ❌")
