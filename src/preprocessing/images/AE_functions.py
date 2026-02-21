import torch
from itertools import combinations
import random
import numpy as np

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

random.seed(42)
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_triplets( labels):
    triplets = []
    labels = labels.cpu().numpy()
    for label in set(labels):
        pos_idx = [i for i in range(len(labels)) if labels[i] == label]
        neg_idx = [i for i in range(len(labels)) if labels[i] != label]
        if len(pos_idx) < 2:
            continue
        for anchor, positive in combinations(pos_idx, 2):
            negative = random.choice(neg_idx)
            triplets.append((anchor, positive, negative))
    return triplets

def extract_embeddings(model, data, device):
    model.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        encoded = model.encoder(data_tensor)
        return encoded.cpu().numpy()

def get_fused_embeddings(model, text_emb, image_emb, device):
    model.to(device)
    model.eval()

    # Unisci embeddings [n, 16]
    combined = np.concatenate([text_emb, image_emb], axis=1)
    combined_tensor = torch.tensor(combined, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(combined_tensor).cpu().numpy()

    return output

def get_single_fused_embeddings(model, text_emb, image_emb, device):
    model.to(device)
    model.eval()

    # Unisci embeddings [n, 16]
    combined = np.concatenate([text_emb, image_emb])
    combined_tensor = torch.tensor(combined, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(combined_tensor).cpu().numpy()

    return output

def normalization(embedding):
    min_val = embedding.min()
    max_val = embedding.max()
    embedding = (embedding - min_val) * (np.pi / (max_val - min_val))
    return embedding

# === Training ===
def train_triplet_autoencoder(model, X, y, n_epochs=100, batch_size=32, lr=1e-3, margin=1.0, weight_mse=1):
    model.to(device)
    criterion_recon = nn.MSELoss()
    criterion_triplet = nn.TripletMarginLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    min_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            emb, xb_recon = model(xb)

            loss_recon = criterion_recon(xb_recon, xb)

            # Triplet loss
            triplets = generate_triplets(yb)
            if triplets:
                anchor = torch.stack([emb[a] for a, _, _ in triplets])
                positive = torch.stack([emb[p] for _, p, _ in triplets])
                negative = torch.stack([emb[n] for _, _, n in triplets])
                loss_triplet = criterion_triplet(anchor, positive, negative)
                loss = (weight_mse * loss_recon) + loss_triplet
            else:
               loss = loss_recon  # fallback solo ricostruzione

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss:.4f}")
    return model


# === Training con Triplet Loss ===
def train_fusion_triplet_model(model, dataset, epochs=50, batch_size=32, lr=1e-3):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for anchor, positive, negative in loader:
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = criterion(anchor_out, positive_out, negative_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
    return model

############################ TRAIN WITH EARLYSTOPPING ##############################àà
# === Training ===
def train_triplet_autoencoder2(model, X, y, n_epochs=100, batch_size=32, lr=1e-3, margin=1.0, weight_mse=1, patience=10):
    model.to(device)
    criterion_recon = nn.MSELoss()
    criterion_triplet = nn.TripletMarginLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            emb, xb_recon = model(xb)

            loss_recon = criterion_recon(xb_recon, xb)

            triplets = generate_triplets(yb)
            if triplets:
                anchor = torch.stack([emb[a] for a, _, _ in triplets])
                positive = torch.stack([emb[p] for _, p, _ in triplets])
                negative = torch.stack([emb[n] for _, _, n in triplets])
                loss_triplet = criterion_triplet(anchor, positive, negative)
                loss = (weight_mse * loss_recon) + loss_triplet
            else:
                loss = loss_recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss:.4f}")

        # Early stopping check
        if total_loss < best_loss - 1e-4:  # tol piccolo per variazioni insignificanti
            best_loss = total_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

# === Training con Triplet Loss ===
def train_fusion_triplet_model2(model, dataset, epochs=50, batch_size=32, lr=1e-3, patience=10):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for anchor, positive, negative in loader:
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = criterion(anchor_out, positive_out, negative_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        # Early stopping
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model









