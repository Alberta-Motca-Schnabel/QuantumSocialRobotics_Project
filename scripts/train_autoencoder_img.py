import sys
import os
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# IMPORT 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.img_autoencoder import TripletAutoencoder_img
import src.preprocessing.AE_functions as ae_funcs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROCESSED_DIR = os.path.join("data", "processed")

# subfolders 
OUT_QUANTUM = os.path.join(PROCESSED_DIR, "quantum_ready")
OUT_CLASSIC = os.path.join(PROCESSED_DIR, "classic_ready")
OUT_RAW = os.path.join(PROCESSED_DIR, "raw_8dim") # Intermediate backup

for d in [OUT_QUANTUM, OUT_CLASSIC, OUT_RAW]:
    os.makedirs(d, exist_ok=True)

# Parameters
SAMPLES_PER_CLASS = 400
EMBEDDING_DIM = 8

def load_data(subset):
    x_path = os.path.join(PROCESSED_DIR, f"X_{subset}_768.npy")
    y_path = os.path.join(PROCESSED_DIR, f"y_{subset}.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Missing files for {subset}!")
    print(f"Loading {subset}")
    return np.load(x_path), np.load(y_path)

def subsample_balanced(X, y, samples_per_class):
    print(f"\nSUBSAMPLING (Train): Target {samples_per_class} per class")
    np.random.seed(42)
    classes = np.unique(y)
    indices_to_keep = []
    
    stats = {}
    for c in classes:
        idx_class = np.where(y == c)[0]
        n_select = min(len(idx_class), samples_per_class)
        selected = np.random.choice(idx_class, n_select, replace=False)
        indices_to_keep.append(selected)
        stats[c] = n_select
        
    final_indices = np.concatenate(indices_to_keep)
    np.random.shuffle(final_indices)
    
    return X[final_indices], y[final_indices]

def normalize_0_1(X_train, X_test):
    """Classic Normalization (MinMax 0-1) fitted on Train"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit only on Train, Transform on both
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # Load Data (768 dim)
    X_train_full, y_train_full = load_data("train")
    X_test, y_test = load_data("test")
    
    # Train Subsampling
    X_train_sub, y_train_sub = subsample_balanced(X_train_full, y_train_full, SAMPLES_PER_CLASS)
    
    # Autoencoder Training
    print("\n STARTING AUTOENCODER TRAINING ")
    model = TripletAutoencoder_img(input_dim=768, bottleneck_dim=EMBEDDING_DIM)
    
    trained_model = ae_funcs.train_triplet_autoencoder2(
        model=model,
        X=X_train_sub,
        y=y_train_sub,
    )
    
    # Extract embeddings (768->8)
    print("\nEMBEDDING EXTRACTION (Reduction 768->8)")
    X_train_8_raw = ae_funcs.extract_embeddings(trained_model, X_train_sub, DEVICE)
    X_test_8_raw = ae_funcs.extract_embeddings(trained_model, X_test, DEVICE)
    
    # Save
    np.save(os.path.join(OUT_RAW, "X_train_8_raw.npy"), X_train_8_raw)
    np.save(os.path.join(OUT_RAW, "y_train.npy"), y_train_sub)
    np.save(os.path.join(OUT_RAW, "X_test_8_raw.npy"), X_test_8_raw)
    np.save(os.path.join(OUT_RAW, "y_test.npy"), y_test)
    print(f"Raw data saved in {OUT_RAW}")

    # Quantum normalization (0 - pi)
    print("\nQUANTUM NORMALIZATION (0 - pi)")
    #function in AE_functions
    X_train_quant = ae_funcs.normalization(X_train_8_raw)
    X_test_quant = ae_funcs.normalization(X_test_8_raw)
    
    np.save(os.path.join(OUT_QUANTUM, "X_train_8_norm.npy"), X_train_quant)
    np.save(os.path.join(OUT_QUANTUM, "y_train.npy"), y_train_sub)
    np.save(os.path.join(OUT_QUANTUM, "X_test_8_norm.npy"), X_test_quant)
    np.save(os.path.join(OUT_QUANTUM, "y_test.npy"), y_test)

    # classic normalization (0 - 1)
    print("\nCLASSIC NORMALIZATION (0 - 1)")
    # standard MinMaxScaler
    X_train_class, X_test_class = normalize_0_1(X_train_8_raw, X_test_8_raw)
    
    np.save(os.path.join(OUT_CLASSIC, "X_train_8_norm.npy"), X_train_class)
    np.save(os.path.join(OUT_CLASSIC, "y_train.npy"), y_train_sub)
    np.save(os.path.join(OUT_CLASSIC, "X_test_8_norm.npy"), X_test_class)
    np.save(os.path.join(OUT_CLASSIC, "y_test.npy"), y_test)
    