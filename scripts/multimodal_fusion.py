import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.images import AE_functions as ae_funcs

# PATH
TXT_PATH = r"C:\Users\Utente\Desktop\QuantumSocialRobotics\data\processed\text\quantum_ready\sampled"
IMG_PATH = r"C:\Users\Utente\Desktop\QuantumSocialRobotics\data\processed\img\quantum_ready"
OUT_PATH = r"C:\Users\Utente\Desktop\QuantumSocialRobotics\data\processed\multimodal\quantum_ready"

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

# DATASET FUSION FUNCTION
def create_fused_dataset(X_txt, y_txt, X_img, y_img, subset_name):
    fused_X = []
    fused_y = []
    
    classes = np.unique(y_img)
    print(f"\nCreating Fused Dataset: {subset_name}")
    
    for c in classes:
        idx_txt = np.where(y_txt == c)[0]
        idx_img = np.where(y_img == c)[0]
        
        if len(idx_txt) == 0 or len(idx_img) == 0:
            print(f"Skipping class {c} - missing data in one modality.")
            continue
            
        n_samples = min(len(idx_txt), len(idx_img))
        
        np.random.seed(42 + c) 
        chosen_txt = np.random.choice(idx_txt, n_samples, replace=False)
        chosen_img = np.random.choice(idx_img, n_samples, replace=False)
        
        for i in range(n_samples):
            fused_vec = np.concatenate([X_txt[chosen_txt[i]], X_img[chosen_img[i]]])
            fused_X.append(fused_vec)
            fused_y.append(c)
            
        print(f"Class {c}: created {n_samples} multimodal pairs.")
            
    fused_X = np.array(fused_X)
    fused_y = np.array(fused_y)
    
    shuffle_idx = np.random.permutation(len(fused_y))
    fused_X = fused_X[shuffle_idx]
    fused_y = fused_y[shuffle_idx]
    
    print(f"-> {subset_name} completed. Shape: X={fused_X.shape}, y={fused_y.shape}")
    return fused_X, fused_y

# TRIPLET DATASET DEFINITION
class TripletFusionDataset(Dataset):
    def __init__(self, combined_emb, labels):
        self.combined = torch.tensor(combined_emb, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.label_to_indices = {int(label): (self.labels == label).nonzero(as_tuple=True)[0]
                                 for label in torch.unique(self.labels)}

    def __len__(self):
        return len(self.combined)

    def __getitem__(self, index):
        anchor = self.combined[index]
        label = self.labels[index].item()

        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label].numpy())
        positive = self.combined[positive_index]

        negative_label = label
        while negative_label == label:
            negative_label = np.random.choice(list(self.label_to_indices.keys()))
        negative_index = np.random.choice(self.label_to_indices[negative_label].numpy())
        negative = self.combined[negative_index]

        return anchor, positive, negative

# 16D -> 8D 
class FusionEmbedding(nn.Module):
    def __init__(self):
        super(FusionEmbedding, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 8)
        )

    def forward(self, x):
        return self.model(x)

# MAIN 
def main():
    print("Loading 8D .npy files (Text and Images)")
    
    X_train_txt = np.load(os.path.join(TXT_PATH, "X_train_sampled.npy"))
    y_train_txt = np.load(os.path.join(TXT_PATH, "y_train_sampled.npy"))
    X_train_img = np.load(os.path.join(IMG_PATH, "X_train_8_norm.npy"))
    y_train_img = np.load(os.path.join(IMG_PATH, "y_train.npy"))

    X_test_txt = np.load(os.path.join(TXT_PATH, "X_test_sampled.npy"))
    y_test_txt = np.load(os.path.join(TXT_PATH, "y_test_sampled.npy"))
    X_test_img = np.load(os.path.join(IMG_PATH, "X_test_8_norm.npy"))
    y_test_img = np.load(os.path.join(IMG_PATH, "y_test.npy"))

    X_train_16D, y_train_fused = create_fused_dataset(X_train_txt, y_train_txt, X_train_img, y_train_img, "TRAIN")
    X_test_16D, y_test_fused = create_fused_dataset(X_test_txt, y_test_txt, X_test_img, y_test_img, "TEST")

    np.save(os.path.join(OUT_PATH, "X_train_16D_fused.npy"), X_train_16D)
    np.save(os.path.join(OUT_PATH, "X_test_16D_fused.npy"), X_test_16D)

    # EXPORTING CSV 
    print("\nSAVING CSV FILES FOR TRANSPARENCY")
    feature_cols = [str(i) for i in range(16)]
    
    train_df = pd.DataFrame(X_train_16D, columns=feature_cols)
    train_df['labels'] = y_train_fused
    train_df.to_csv(os.path.join(OUT_PATH, "multimodal_train_16D_with_labels.csv"), index=False)
    
    test_df = pd.DataFrame(X_test_16D, columns=feature_cols)
    test_df['labels'] = y_test_fused
    test_df.to_csv(os.path.join(OUT_PATH, "multimodal_test_16D_with_labels.csv"), index=False)
    print("CSV files saved successfully.")

    # Train Fusion Autoencoder using AE_functions
    print("\nSTARTING FUSION AUTOENCODER TRAINING (16D -> 8D)")
    
    train_dataset = TripletFusionDataset(X_train_16D, y_train_fused)
    model = FusionEmbedding()
    
    # Using function with Early Stopping
    trained_model = ae_funcs.train_fusion_triplet_model2(
        model=model, 
        dataset=train_dataset, 
        epochs=100,      
        batch_size=32, 
        lr=1e-3, 
        patience=10
    )

    torch.save(trained_model.state_dict(), os.path.join(OUT_PATH, "fusion_model_weights.pt"))
    print("Training completed. Weights saved.")

    # Extract 8D Embeddings
    print("\nEXTRACTING 8D EMBEDDINGS")
    trained_model.eval()
    with torch.no_grad():
        X_train_8D = trained_model(torch.tensor(X_train_16D, dtype=torch.float32)).numpy()
        X_test_8D = trained_model(torch.tensor(X_test_16D, dtype=torch.float32)).numpy()

    # Normalization
    print("QUANTUM NORMALIZATION (0 - pi)")
    X_train_8_norm = ae_funcs.normalization(X_train_8D)
    X_test_8_norm = ae_funcs.normalization(X_test_8D)

    # Save
    np.save(os.path.join(OUT_PATH, "X_train_8_norm.npy"), X_train_8_norm)
    np.save(os.path.join(OUT_PATH, "X_test_8_norm.npy"), X_test_8_norm)
    np.save(os.path.join(OUT_PATH, "y_train.npy"), y_train_fused)
    np.save(os.path.join(OUT_PATH, "y_test.npy"), y_test_fused)

    print("\nMULTIMODAL PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Final Train set: X={X_train_8_norm.shape}, y={y_train_fused.shape}")
    print(f"Final Test set:  X={X_test_8_norm.shape}, y={y_test_fused.shape}")

if __name__ == "__main__":
    main()