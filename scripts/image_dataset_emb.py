import sys
import os
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.EmbeddingsExtraction import EmbeddingsExtraction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATASET_ROOT = os.path.join("data", "raw", "FER2013") 
PROCESSED_DIR = os.path.join("data", "processed")
REPORTS_DIR = os.path.join("reports")

# Create output directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def get_image_paths(subset_dir):
    """Retrieves image paths and generates numeric labels."""
    if not os.path.exists(subset_dir):
        return [], [], {}
        
    image_paths = []
    labels = []
    
    # Sort classes for consistent mapping 
    class_names = sorted([d for d in os.listdir(subset_dir) if os.path.isdir(os.path.join(subset_dir, d))])
    label_map = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(subset_dir, class_name)
        valid_exts = ('.jpg')
        
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(valid_exts):
                image_paths.append(os.path.join(class_path, img_name))
                labels.append(label_map[class_name])
                
    return image_paths, np.array(labels), label_map

def extract_and_save(subset_name, paths, labels, extractor):
    """Extracts embeddings (768-dim) and saves .npy files."""
    print(f"\nProcessing set: {subset_name.upper()} ({len(paths)} images)")
    
    npy_path = os.path.join(PROCESSED_DIR, f"X_{subset_name}_768.npy")
    lbl_path = os.path.join(PROCESSED_DIR, f"y_{subset_name}.npy")
    
    # Check for existing files to skip reprocessing
    if os.path.exists(npy_path) and os.path.exists(lbl_path):
        print(f"Files found in {PROCESSED_DIR}. Loading")
        return np.load(npy_path), np.load(lbl_path)

    embeddings = []
    valid_labels = []
    
    # Extraction loop 
    for i, img_path in enumerate(tqdm(paths, desc=f"Extracting {subset_name}")):
        try:
            emb = extractor.extract_Img_Emb(img_path, device=DEVICE)
            embeddings.append(emb.cpu().numpy())
            valid_labels.append(labels[i])
        except Exception as e:
            print(f"\nError on {img_path}: {e}")

    X = np.stack(embeddings)
    y = np.array(valid_labels)
    
    print(f"Saving to {PROCESSED_DIR}")
    np.save(npy_path, X)
    np.save(lbl_path, y)
    
    return X, y

def generate_analysis_report(y_train, y_test, class_map):
    """Generates a detailed text report in the reports/ folder."""
    inv_map = {v: k for k, v in class_map.items()}
    report_path = os.path.join(REPORTS_DIR, "01_dataset_analysis.txt")
    
    total_train = len(y_train)
    total_test = len(y_test)
    total_global = total_train + total_test
    
    with open(report_path, "w") as f:
        f.write("=== FER2013 DATASET ANALYSIS REPORT ===\n")
        f.write(f"Total classes: {len(class_map)}\n")
        f.write(f"Mapping: {class_map}\n\n")
        
        # Table Header
        header = f"{'CLASS':<15} | {'TRAIN':<10} | {'TEST':<10} | {'TOTAL':<10} | {'% GLOBAL'}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        unique_classes = sorted(class_map.values())
        for cls_idx in unique_classes:
            cls_name = inv_map[cls_idx]
            
            n_train = np.sum(y_train == cls_idx)
            n_test = np.sum(y_test == cls_idx)
            n_tot = n_train + n_test
            perc = (n_tot / total_global * 100) if total_global > 0 else 0
            
            row = f"{cls_name:<15} | {n_train:<10} | {n_test:<10} | {n_tot:<10} | {perc:.2f}%"
            f.write(row + "\n")
            
        f.write("-" * len(header) + "\n")
        f.write(f"{'TOTALS':<15} | {total_train:<10} | {total_test:<10} | {total_global:<10} | 100%\n")

    print(f"\nCheck file: {report_path}")

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # Initialize Extractor
    extractor = EmbeddingsExtraction()
    
    # Process TRAIN
    train_dir = os.path.join(DATASET_ROOT, "train")
    paths_train, labels_train, map_train = get_image_paths(train_dir)
    
    if not paths_train:
        print(f"No images found in {train_dir}. Check path.")
        sys.exit(1)
        
    X_train, y_train = extract_and_save("train", paths_train, labels_train, extractor)

    # Process TEST
    test_dir = os.path.join(DATASET_ROOT, "test")
    paths_test, labels_test, _ = get_image_paths(test_dir)
    
    if paths_test:
        X_test, y_test = extract_and_save("test", paths_test, labels_test, extractor)
    else:
        print(f"TEST folder not found in {test_dir}. Partial analysis.")
        X_test, y_test = np.array([]), np.array([])

    # Report
    generate_analysis_report(y_train, y_test, map_train)
    
    print("Embeddings (768-dim) saved in 'data/processed'.")