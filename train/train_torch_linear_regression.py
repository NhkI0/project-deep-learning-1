import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from models.pytorch.torch_linear_regression import LinearRegression
from data.utils.data_loader import load_data

# Variables
LEARNING_RATE = 0.01
EPOCHS = 500

# Chargement des données
df = load_data()

target_cols = ["Height_(cm)", "Weight_(kg)"]
y_raw = df[target_cols].values
x_raw = df.drop(columns=target_cols).values

# Process de normalization
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_scaled = scaler_x.fit_transform(x_raw)
y_scaled = scaler_y.fit_transform(y_raw)

# Conversion en tenseurs pytroch
x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)  # Configuration des dimensions

input_dim = x_tensor.shape[1]
output_dim = y_tensor.shape[1]

print(f"Features en entrée : {input_dim}")
print(f"Valeurs à prédire : {output_dim}")

# Initialisation du model
model = LinearRegression(input_dim, output_dim)

# Gestion des hyperparamètres
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Boucle d'entrainement
print("Début de l'entraînement...")
for epoch in range(EPOCHS):
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item():.4f}")

# Test
sample_input = x_tensor[0]
sample_true_values = y_tensor[0]

with torch.no_grad():
    prediction = model(sample_input)

print(f"Prédiction -> Taille: {prediction[0]:.1f} cm, Poids: {prediction[1]:.1f} kg")
print(
    f"Réalité    -> Taille: {sample_true_values[0]:.1f} cm, Poids: {sample_true_values[1]:.1f} kg"
)
