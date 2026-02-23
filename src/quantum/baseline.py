import numpy as np
import pandas as pd
import time
import os
import argparse
import csv


import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.stats import entropy
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from circuits import (
    encoding_features_HRx,
    encoding_features_HRyRzCnot,
    construct_tensor_ring_ansatz_circuit,
    tensor_ring,
    construct_qnn,
)

# CONFIGURATION 
NUM_QUBITS = 8      
NUM_CLASSES = 7     
MAX_ITER = 1000     

# VISUALIZATION 
def save_circuit_diagram(circuit, name, folder):
    circuit_dir = os.path.join(folder, "circuit_diagrams")
    if not os.path.exists(circuit_dir):
        os.makedirs(circuit_dir)
        
    filename = os.path.join(circuit_dir, f"{name}.png")
    print(f"Saving diagram: {name}")

    try:
        circuit.decompose().draw(output='mpl', style='iqp', fold=-1, filename=filename)
    except Exception as e:
        print(f"Decomposed draw failed, trying normal draw: {e}")
        circuit.draw(output='mpl', style='iqp', fold=-1, filename=filename)

# BUILD ANSATZ
def build_exact_ansatz(ansatz_type, num_qubits):
    if ansatz_type == "MPS_TTN":
        print(f" Building ansatz (Tensor Ring + TTN)")
        return construct_tensor_ring_ansatz_circuit(num_qubits)
    elif ansatz_type == "TensorRing":
        print(f" Building Tensor Ring ansatz")
        return tensor_ring(num_qubits, reps=1)
    else:
        raise ValueError(f"Ansatz type {ansatz_type} not recognized.")

# CALCULATE EXPRESSIVITY 
def calculate_expressivity(ansatz, num_qubits, num_samples=1000):
    print(f"Calculating Expressivity on {num_samples} samples...")
    N = 2**num_qubits 
    p_values = [] 
    for _ in range(num_samples):
        rand_params = np.random.uniform(0, 2*np.pi, ansatz.num_parameters)
        bound_qc = ansatz.assign_parameters(rand_params)
        sv = Statevector(bound_qc)
        probs = np.abs(sv.data)**2
        probs = probs[probs > 1e-10] 
        p_values.extend(N * probs)
    
    p_values = np.array(p_values)
    hist_range = (0, np.max(p_values))
    P_q, bin_edges = np.histogram(p_values, bins=75, range=hist_range, density=True)
    
    P_haar = []
    for i in range(len(bin_edges)-1):
        a, b = bin_edges[i], bin_edges[i+1]
        P_haar.append(np.exp(-a) - np.exp(-b))
        
    P_q = P_q / np.sum(P_q)
    P_haar = np.array(P_haar) / np.sum(P_haar)
    
    kl_div = entropy(P_q, P_haar + 1e-10)
    return kl_div

# METRICS HELPER
def calculate_metrics(y_true, y_pred):
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true

    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_indices = np.argmax(y_pred, axis=1)
    else:
        y_pred_indices = y_pred
    
    acc = accuracy_score(y_true_indices, y_pred_indices)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_indices, y_pred_indices, average='weighted', zero_division=0
    )
    return acc, precision, recall, f1

# PIPELINE 
def run_pipeline(csv_path):
    print(f"{'='*50}\nSTARTING QML PIPELINE\n{'='*50}")
    
    # load data 
    try:
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        

        X = df.iloc[:, :-1].values
        y_raw = df.iloc[:, -1].values
        
        # map strings to numbers
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        
        print(f"Classi trovate: {le.classes_} (Mappate in: {np.unique(y)})")
        
        # Split 80% Train, 20% Test 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Data loaded and split. Train size: {len(X_train)} | Test size: {len(X_test)}")
        print(f"Features shape: {X_train.shape[1]} (Should be {NUM_QUBITS} for {NUM_QUBITS} qubits)")
        
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while processing the CSV: {e}")
        return

    configurations = [
        ("Config_A", encoding_features_HRx, "MPS_TTN"),
        ("Config_B", encoding_features_HRx, "TensorRing"),
        ("Config_C", encoding_features_HRyRzCnot, "MPS_TTN"),
        ("Config_D", encoding_features_HRyRzCnot, "TensorRing")
    ]
    
    results = {}

    for name, fm_func, ansatz_type in configurations:
        print(f"\n{'-'*50}\n Processing: {name}\n{'-'*50}")
        config_start_time = time.time()
        
        try:
            # Build Circuit
            fm = fm_func(NUM_QUBITS)
            ansatz = build_exact_ansatz(ansatz_type, NUM_QUBITS)
            
            # VISUALIZATION 
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            config_plot_dir = os.path.join("plots", base_name, name)
            if not os.path.exists(config_plot_dir):
                os.makedirs(config_plot_dir)
                
            save_circuit_diagram(fm, "FeatureMap", config_plot_dir)
            save_circuit_diagram(ansatz, "Ansatz", config_plot_dir)
            full_circuit = fm.compose(ansatz)
            save_circuit_diagram(full_circuit, "Full_VQC_Circuit", config_plot_dir)

            # Expressivity
            expr_val = calculate_expressivity(ansatz, NUM_QUBITS)
            print(f"Expressivity (KL Divergence): {expr_val:.4f}")

            # Construct VQC 
            vqc = construct_qnn(
                feature_map=fm, 
                ansatz=ansatz, 
                callback_graph=None, 
                maxiter=MAX_ITER,
                output_shape=NUM_CLASSES 
            )
                    
            # Training
            print(f"\n Training VQC... (MaxIter={MAX_ITER}, Classes={NUM_CLASSES})")
            start_train_time = time.time()
            vqc.fit(X_train, y_train) 
            train_duration = time.time() - start_train_time
            print(f"Training completed in {train_duration:.2f}s")
            
            # metrics on train set
            y_train_pred = vqc.predict(X_train)
            train_acc, train_prec, train_rec, train_f1 = calculate_metrics(y_train, y_train_pred)
            
            # metrics in test set
            y_test_pred = vqc.predict(X_test)
            test_acc, test_prec, test_rec, test_f1 = calculate_metrics(y_test, y_test_pred)
            
            print("\n --- TRAIN METRICS ---")
            print(f"   Accuracy:  {train_acc:.4f} | Precision: {train_prec:.4f} | Recall: {train_rec:.4f} | F1: {train_f1:.4f}")
            print(" --- TEST METRICS  ---")
            print(f"   Accuracy:  {test_acc:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1: {test_f1:.4f}")
            
            # Save Weights
            np.save(f"weights_{base_name}_{name}.npy", vqc.weights)
            
            config_total_time = time.time() - config_start_time

            results[name] = {
                "train_acc": train_acc, "train_prec": train_prec, "train_rec": train_rec, "train_f1": train_f1,
                "test_acc": test_acc,   "test_prec": test_prec,   "test_rec": test_rec,   "test_f1": test_f1,
                "status": "Success", "expressivity": expr_val, "duration": config_total_time 
            }
            
        except Exception as e:
            print(f"\n ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"status": "Error"}

        
    output_csv_file = f"VQC_results_{base_name}.csv"
    csv_data = []
    
    for name, res in results.items():
        if res["status"] == "Success":
             csv_data.append({
                 "Configuration": name,
                 "Train_Accuracy": round(res['train_acc'], 4),
                 "Train_Precision": round(res['train_prec'], 4),
                 "Train_Recall": round(res['train_rec'], 4),
                 "Train_F1": round(res['train_f1'], 4),
                 "Test_Accuracy": round(res['test_acc'], 4),
                 "Test_Precision": round(res['test_prec'], 4),
                 "Test_Recall": round(res['test_rec'], 4),
                 "Test_F1": round(res['test_f1'], 4),
                 "Expressivity": round(res['expressivity'], 4),
                 "Execution_Time_sec": round(res['duration'], 2),
                 "Status": res['status']
             })
        else:
             csv_data.append({
                 "Configuration": name,
                 "Train_Accuracy": "", "Train_Precision": "", "Train_Recall": "", "Train_F1": "",
                 "Test_Accuracy": "", "Test_Precision": "", "Test_Recall": "", "Test_F1": "",
                 "Expressivity": "", "Execution_Time_sec": "", "Status": "Failed"
             })

    # write on csv
    keys = ["Configuration", "Train_Accuracy", "Train_Precision", "Train_Recall", "Train_F1", 
            "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", 
            "Expressivity", "Execution_Time_sec", "Status"]
    
    with open(output_csv_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(csv_data)
        
    print(f"\n[{time.strftime('%H:%M:%S')}] Pipeline Completata! Risultati salvati nel file: {output_csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QML Pipeline on a CSV dataset.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    
    args = parser.parse_args()
    
    run_pipeline(args.csv_path)