import numpy as np
import time
import os
import sys

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.stats import entropy
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# circuits.py
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
    print(f"Calculating on {num_samples} samples")
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

# PIPELINE 
def run_pipeline():
    print(f"STARTING QML PIPELINE")
    
    base_path = r"C:\Users\Utente\Desktop\QuantumSocialRobotics\data\processed\img\quantum_ready"
    
    try:
        print("Loading data")
        X_train = np.load(os.path.join(base_path, "X_train_8_norm.npy"))
        X_test  = np.load(os.path.join(base_path, "X_test_8_norm.npy"))
        y_train = np.load(os.path.join(base_path, "y_train.npy"))
        y_test  = np.load(os.path.join(base_path, "y_test.npy"))
        print(f"Data loaded. Train: {len(X_train)} | Test: {len(X_test)}")
    except FileNotFoundError:
        print(f"Files not found in {base_path}. Check paths.")
        return

    configurations = [
        ("Config_A", encoding_features_HRx, "MPS_TTN"),
        ("Config_B", encoding_features_HRx, "TensorRing"),
        ("Config_C", encoding_features_HRyRzCnot, "MPS_TTN"),
        ("Config_D", encoding_features_HRyRzCnot, "TensorRing")
    ]
    
    results = {}

    for name, fm_func, ansatz_type in configurations:
        print(f"\n Processing: {name}")
        config_start_time = time.time()
        
        try:
            # Build Circuit
            fm = fm_func(NUM_QUBITS)
            ansatz = build_exact_ansatz(ansatz_type, NUM_QUBITS)
            
            # VISUALIZATION 
            config_plot_dir = os.path.join("plots", name)
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
            print(f"\n MaxIter={MAX_ITER}, Classes={NUM_CLASSES}")
            start_train_time = time.time()
            vqc.fit(X_train, y_train) 
            train_duration = time.time() - start_train_time
            print(f"Training completed in {train_duration:.2f}s")
            
            # Evaluation
            y_pred = vqc.predict(X_test)
            
            # Metrics Conversion
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                y_true_indices = np.argmax(y_test, axis=1)
            else:
                y_true_indices = y_test

            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred_indices = np.argmax(y_pred, axis=1)
            else:
                y_pred_indices = y_pred
            
            acc = accuracy_score(y_true_indices, y_pred_indices)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_indices, y_pred_indices, average='weighted', zero_division=0
            )
            
            print(f"   -> Accuracy:  {acc:.4f}")
            print(f"   -> Precision: {precision:.4f}")
            print(f"   -> Recall:    {recall:.4f}")
            print(f"   -> F1 Score:  {f1:.4f}")
            
            # Save Weights
            np.save(f"weights_{name}.npy", vqc.weights)
            
            config_total_time = time.time() - config_start_time

            results[name] = {
                "acc": acc, "prec": precision, "rec": recall, "f1": f1, 
                "status": "Success", "expressivity": expr_val, "duration": config_total_time 
            }
            
        except Exception as e:
            print(f"\n ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"status": "Error"}

    print("\nCOMPLETE FINAL RESULTS ")
    print(f"{'Configuration':<15} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'Expr':<8} | {'Time(s)':<8}")
    print("-" * 90)
    
    # CSV 
    import csv
    csv_file = "VQC_image_results.csv"
    csv_data = []
    
    for name, res in results.items():
        if res["status"] == "Success":
             print(f"{name:<15} | {res['acc']:.4f}   | {res['prec']:.4f}   | {res['rec']:.4f}   | {res['f1']:.4f}   | {res['expressivity']:.4f}   | {res['duration']:.2f}")
             
             csv_data.append({
                 "Configuration": name,
                 "Accuracy": round(res['acc'], 4),
                 "Precision": round(res['prec'], 4),
                 "Recall": round(res['rec'], 4),
                 "F1_Score": round(res['f1'], 4),
                 "Expressivity": round(res['expressivity'], 4),
                 "Execution_Time_sec": round(res['duration'], 2),
                 "Status": res['status']
             })
        else:
             print(f"{name:<15} | FAILED")
             csv_data.append({
                 "Configuration": name,
                 "Accuracy": "", "Precision": "", "Recall": "", "F1_Score": "", "Expressivity": "", "Execution_Time_sec": "",
                 "Status": "Failed"
             })

    # writing of the CSV file
    keys = ["Configuration", "Accuracy", "Precision", "Recall", "F1_Score", "Expressivity", "Execution_Time_sec", "Status"]
    with open(csv_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(csv_data)
        
    print(f"\n Results saved in the file: {csv_file}")

if __name__ == "__main__":
    run_pipeline()