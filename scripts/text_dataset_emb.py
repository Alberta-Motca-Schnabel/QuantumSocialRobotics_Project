import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import os
import sys
from sklearn.model_selection import train_test_split

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modules
from src.preprocessing.EmbeddingsExtraction import EmbeddingsExtraction
from src.preprocessing.text.txt_autoencoder import TripletAutoencoder_txt
from src.preprocessing.images import AE_functions as ae_funcs
from src.preprocessing.text.Utility import encoder_labels 

torch.set_num_threads(4) 

CSV_PATH = "data/processed/text/Text_Dataset_Ekman_Tokenized.csv"
OUTPUT_DIR = "data/processed/text/quantum_ready" 
TEMP_FILE = "data/processed/text/X_all_768_TEMP.pt" 

TEXT_COLUMN = "text"   
LABEL_COLUMN = "labels" 

SAMPLES_PER_CLASS_TRAIN = 400

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# SUBSAMPLINGTR TRAIN
def subsample_balanced(X, y, samples_per_class):
    print(f"\nSUBSAMPLING (Train): Target {samples_per_class} per class")
    np.random.seed(42)
    classes = np.unique(y)
    indices_to_keep = []
    
    for c in classes:
        idx_class = np.where(y == c)[0]
        n_select = min(len(idx_class), samples_per_class)
        selected = np.random.choice(idx_class, n_select, replace=False)
        indices_to_keep.append(selected)
        
    final_indices = np.concatenate(indices_to_keep)
    np.random.shuffle(final_indices)
    
    return X[final_indices], y[final_indices]

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    print(f"Loading dataset from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=[TEXT_COLUMN])
    texts = df[TEXT_COLUMN].astype(str).tolist()
    labels_str = df[LABEL_COLUMN].values
    
    labels_encoded = np.array(encoder_labels(labels_str))

    # LOADING OF 768D EMBEDDINGS 
    if os.path.exists(TEMP_FILE):
        print(f"Temporary file found! Loading embeddings from {TEMP_FILE}")
        X_all = torch.load(TEMP_FILE)
    else:
        print("Initializing models for extraction")
        extractor = EmbeddingsExtraction()
        extractor.modelText = extractor.modelText.to(DEVICE)
        
        embeddings_list = []
        for text in tqdm(texts):
            emb = extractor.extract_Text_Emb(text, device=DEVICE)
            embeddings_list.append(emb.cpu())
        X_all = torch.stack(embeddings_list)
        torch.save(X_all, TEMP_FILE)

    X_all_np = X_all.numpy()

    # 80/20 SPLIT
    print("\n Train/Test split (80/20)")
    X_train_768_full, X_test_768_full, y_train_full, y_test_full = train_test_split(
        X_all_np, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
    )

    # SUBSAMPLING TRAIN ONLY, KEEPING FULL TEST SET
    X_train_768_sub, y_train_sub = subsample_balanced(X_train_768_full, y_train_full, SAMPLES_PER_CLASS_TRAIN)
    X_test_768_final = X_test_768_full
    y_test_final = y_test_full

    # TRAINING AUTOENCODER ONLY ON SAMPLED TRAIN DATA
    print("\nAUTOENCODER TRAINING")
    model = TripletAutoencoder_txt(input_dim=768, bottleneck_dim=8).to(DEVICE)
    
    trained_model = ae_funcs.train_triplet_autoencoder2(
        model=model,
        X=X_train_768_sub,
        y=y_train_sub,
    )

    # 8D EXTRACTION
    print("\nEMBEDDING EXTRACTION (Reduction 768->8)")
    X_train_8_raw = ae_funcs.extract_embeddings(trained_model, X_train_768_sub, DEVICE)
    X_test_8_raw = ae_funcs.extract_embeddings(trained_model, X_test_768_final, DEVICE)

    # NORMALIZATION 
    print("QUANTUM NORMALIZATION (0 - pi)")
    X_train_8_norm = ae_funcs.normalization(X_train_8_raw)
    X_test_8_norm = ae_funcs.normalization(X_test_8_raw)

    # SAVING 
    print(f"Saving files in {OUTPUT_DIR}")
    np.save(os.path.join(OUTPUT_DIR, "X_train_8_norm.npy"), X_train_8_norm)
    np.save(os.path.join(OUTPUT_DIR, "X_test_8_norm.npy"), X_test_8_norm)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train_sub)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test_final)


    print(f"Train set: X={X_train_8_norm.shape}, y={y_train_sub.shape}")
    print(f"Test set:  X={X_test_8_norm.shape}, y={y_test_final.shape}")

if __name__ == '__main__':
    main()