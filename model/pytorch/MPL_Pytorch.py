import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from MultiOutputNN import MultiOutputNN

torch.manual_seed(42)

BATCH_SIZE = 64
EPOCHS = 20
VALID_LOSSES = []

def load_and_preprocess_data():
    """Charger les données et preprocess des datas"""
    df = pd.read_csv("../../data/CVD_cleaned_dummies.csv")

    target_columns = ["Heart_Disease_Yes", "Skin_Cancer_Yes", "Other_Cancer_Yes", "Arthritis_Yes"]

    X = df.drop(target_columns, axis=1, errors='ignore') #On retire les columns cible du dataset
    y = df[target_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    return X_train_scaled_df, X_test_scaled_df, y_train_scaled, y_test_scaled, scaler_y

def get_best_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps")

def training():
    for epoch in range(EPOCHS):
        start_time = time.time() # Pour avoir le temps de run total
        model.train()

        for batch_id, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)

            loss = criterion(y_pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            elapsed_time = time.time() - start_time

            print(f"\rEpochs:{epoch+1} Batch:{batch_id+1} Loss:{loss.item():.4f} Time:{elapsed_time:.2f} sec", end="")


if __name__ == "__main__":

    X_train, X_test, y_train, y_test, scaler_y = load_and_preprocess_data()

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    val_size = int(0.2 * len(X_train_tensor))
    train_size = len(X_train_tensor) - val_size

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    # Récupération du meilleur device
    device = get_best_device()

    # Chargement du model
    model = MultiOutputNN(input_dim, output_dim).to(device)

    # Hyperparamètre
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Lancement de l'entrainement
    training()

    model.eval()
    valid_loss = 0

    with torch.no_grad():
        for batch_id, (batch_x, batch_y) in enumerate(val_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            valid_loss += loss.item()

    valid_loss /= len(val_loader)
    VALID_LOSSES.append(valid_loss)

    print(f" Valid loss:{valid_loss:.4f}")
