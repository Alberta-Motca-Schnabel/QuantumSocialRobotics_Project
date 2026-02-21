import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

# CONFIGURATION & PATHS
SEED = 42 # seed for reproducibility
DATA_DIR = r"C:\Users\Utente\Desktop\QuantumSocialRobotics\data\processed\img\classic_ready"
RESULTS_DIR = "reports/final_experiment"  # Output folder
BATCH_SIZE = 32
EPOCHS = 150
PATIENCE = 12
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# MODEL DEFINITIONS
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=8, num_classes=7):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, input_dim=8, num_classes=7):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dimension (batch, 1, 8)
        x = self.features(x)
        x = self.classifier(x)
        return x

# PLOTS AND REPORTS
def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

def save_loss_plot(train_losses, val_losses, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Dynamics')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

# TRAINING EXECUTION
def train_and_evaluate(model_type, X_train_full, y_train_full, X_test, y_test):
    set_seed(SEED)
    
    # Train/Validation Split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=SEED, stratify=y_train_full
    )

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=BATCH_SIZE, shuffle=False)

    # Init Model
    if model_type == 'mlp':
        model = SimpleMLP().to(DEVICE)
    else:
        model = SimpleCNN().to(DEVICE)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = model.state_dict() # initialize best state
    train_history, val_history = [], []
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        # Validation
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                v_loss += criterion(model(X_b), y_b).item()
        
        avg_train = t_loss / len(train_loader)
        avg_val = v_loss / len(val_loader)
        train_history.append(avg_train)
        val_history.append(avg_val)
        
        # Early Stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # FINAL TEST
    model.load_state_dict(best_model_state)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            outputs = model(X_b)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(y_b.cpu().numpy())
            
    # Calculate Metrics
    metrics = {
        'accuracy': accuracy_score(targets, preds),
        'f1_macro': f1_score(targets, preds, average='macro'),
        'f1_weighted': f1_score(targets, preds, average='weighted', zero_division=0),
        'precision': precision_score(targets, preds, average='weighted', zero_division=0),
        'recall': recall_score(targets, preds, average='weighted', zero_division=0),
        'loss_history': (train_history, val_history),
        'predictions': preds,
        'targets': targets
    }
    
    return metrics

# MAIN
if __name__ == "__main__":
    print(f"Loading data")
    try:
        X_train_full = np.load(os.path.join(DATA_DIR, "X_train_8_norm.npy"))
        y_train_full = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        X_test = np.load(os.path.join(DATA_DIR, "X_test_8_norm.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    except Exception as e:
        print(f"error {e}")
        sys.exit(1)

# Choose 'cnn' or 'mlp'
    MODEL_TYPE = 'cnn'  
    
    print(f"[START] Training {MODEL_TYPE.upper()} model")
    res = train_and_evaluate(MODEL_TYPE, X_train_full, y_train_full, X_test, y_test)
    
    # Save plots
    save_loss_plot(res['loss_history'][0], res['loss_history'][1], f"{MODEL_TYPE}_loss_curve.png")
    save_confusion_matrix(res['targets'], res['predictions'], 
                          f"Confusion Matrix ({MODEL_TYPE.upper()})", f"{MODEL_TYPE}_confusion_matrix.png")
    
    # Generate Text Report
    stats_path = os.path.join(RESULTS_DIR, f"{MODEL_TYPE}_final_report.txt")
    with open(stats_path, "w") as f:
        f.write(f"=== FINAL REPORT ({MODEL_TYPE.upper()}) ===\n\n")
        f.write(f"Accuracy:    {res['accuracy']:.4f}\n")
        f.write(f"Precision:   {res['precision']:.4f}\n")
        f.write(f"Recall:      {res['recall']:.4f}\n")
        f.write(f"F1 Weighted: {res['f1_weighted']:.4f}\n")
        f.write(f"F1 Macro:    {res['f1_macro']:.4f}\n\n")
        f.write("DETAILED CLASSIFICATION REPORT \n")
        f.write(classification_report(res['targets'], res['predictions'], target_names=CLASS_NAMES, zero_division=0))

    print("\n" + "="*50)
    print(f"[COMPLETED] Results saved in: {RESULTS_DIR}")
    print(f"Accuracy:    {res['accuracy']*100:.2f}%")
    print(f"F1 Weighted: {res['f1_weighted']:.4f}")
    print("="*50)