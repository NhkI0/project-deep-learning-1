import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix

from models.pytorch.torch_classification import Classification
from data.utils.data_loader import load_data

# Variables
LEARNING_RATE = 0.001
EPOCHS = 200

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Chargement du dataset
df = load_data()

# Définir la target
target_col = "Heart_Disease_Yes"

# Séparation des features
y_numpy = df[target_col].values
x_numpy = df.drop(columns=target_col).values

# Split train
X_train, X_test, y_train, y_test = train_test_split(
    x_numpy, y_numpy, test_size=0.2, random_state=42, stratify=y_numpy
)

# SMOTE
print(f"Avant : {sum(y_train == 1)} malades sur {len(y_train)} total")

smote = SMOTE(sampling_strategy=0.8, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(
    f"Après : {sum(y_train_resampled == 1)} malades sur {len(y_train_resampled)} total (Équilibre 50/50)"
)

# Scale des données
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Conversion en tenseur pytorch
x_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = (
    torch.tensor(y_train_resampled, dtype=torch.float32).reshape(-1, 1).to(device)
)

x_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)

# Initiation du model
input_dim = x_train_tensor.shape[1]

model = Classification(input_dim, hidden_layers=[256, 128, 64]).to(device)

# Configuration des hyperparamètres
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Boucle d'entrainement
for epoch in range(EPOCHS):
    model.train()

    logits = model(x_train_tensor)
    loss = criterion(logits, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.3532).float()
            acc = (preds == y_train_tensor).float().mean()
            f1 = f1_score(y_train_tensor.cpu(), preds.cpu())

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] | Loss: {loss.item():.4f} | Acc: {acc:.2f} | F1: {f1:.4f}"
            )


print("\n--- Évaluation TEST ---")
model.eval()
with torch.no_grad():
    test_logits = model(x_test_tensor)
    test_probs = torch.sigmoid(test_logits)
    test_preds = (test_probs > 0.5).float()

    test_acc = (test_preds == y_test_tensor).float().mean()
    test_f1 = f1_score(y_test_tensor.cpu(), test_preds.cpu())

print(f"Précision Test : {test_acc.item() * 100:.2f}%")
print(f"F1 Score Test  : {test_f1:.4f}")

# ==========================================
# RÉSULTATS DÉFINITIFS (PRODUCTION)
# ==========================================

# 1. On fixe le seuil calibré précédemment
FINAL_THRESHOLD = 0.3532

print("\n" + "=" * 40)
print(f"     EVALUATION FINALE (Seuil : {FINAL_THRESHOLD})")
print("=" * 40)

# 2. Inférence sur le jeu de Test
model.eval()
with torch.no_grad():
    logits = model(x_test_tensor)
    probs = torch.sigmoid(logits)  # Probabilités brutes

    # Application du seuil dur
    final_preds = (probs > FINAL_THRESHOLD).float()

    # On récupère les vraies valeurs pour comparer
    y_true = y_test_tensor.cpu().numpy()
    y_pred = final_preds.cpu().numpy()

# 3. Calcul de la Matrice de Confusion
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# 4. Affichage Propre
print(f"Vrais Négatifs (Sains corrects)    : {tn}")
print(f"Faux Positifs  (Fausses alertes)   : {fp}")
print(f"Faux Négatifs  (Malades ratés)     : {fn}")
print(f"Vrais Positifs (Malades détectés)  : {tp}")

print("-" * 40)
print(f"Total Malades dans le Test Set     : {tp + fn}")
print(f"Malades trouvés par le modèle      : {tp}")

# 5. Métriques Finales
recall = tp / (tp + fn)
precision = tp / (tp + fp)

print("-" * 40)
print(f"RECALL : {recall * 100:.2f}%")
print(f"PRÉCISION FINAL            : {precision * 100:.2f}%")

if not os.path.exists("models/pytorch/saved_models"):
    os.makedirs("models/pytorch/saved_models")

torch.save(model.state_dict(), "models/pytorch/saved_models/heart_disease.pth")
print("Modèle sauvegardé dans models/pytorch/saved_models")
