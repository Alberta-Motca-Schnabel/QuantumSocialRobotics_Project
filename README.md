# QuantumSocialRobotics_Project

This project explores emotion recognition using **FER13** (images) and **GoEmotions** (text) datasets. It implements a full pipeline from embedding
 extraction and multimodal fusion to classification using both Classical Machine Learning and VQC.

---

## 1. Preprocessing
The preprocessing pipeline handles feature extraction, dimensionality reduction, and normalization to prepare data for both classical and quantum 
architectures.

* **Scripts:**
    * `scripts/image_dataset_emb.py`: Initial extraction of image embeddings.
    * `scripts/train_autoencoder_img.py`: Dimensionality reduction to 8 embeddings, dataset subsampling, and normalization.
    * `scripts/text_dataset_emb.py`: Performs embedding extraction, reduction, subsampling, and normalization for text data.
    * `scripts/multimodal_fusion.py`: Performs image and text embedding fusion, dimensionality reduction to 8, and final normalization.
* **Core Logic:** Utility functions are stored in `src/preprocessing/`.
* **Data Storage:** Processed datasets are located in the `data/processed/` folder, which includes:
    * `quantum_ready/`: Embeddings normalized for quantum training.
    * `classic_ready/`: Embeddings normalized for classic ML training.

---

## 2. Classical ML

* **Implementation:** `src/classic/CNN_MLP.py`
* **Usage:** This script performs both 1D CNN and MLP training. It is possible to switch between them in the `main` section by specifying:
    * `MODEL_TYPE = 'cnn'`
    * `MODEL_TYPE = 'mlp'`
* **Outputs:** Performance reports and generated confusion matrices are saved in the `results/` folder.

---

## 3. Quantum ML
Implementation of VQC using the feature maps required and the provided ansatz.

* **Circuit Definitions:** `src/quantum/circuits.py` contains the two used Ansatz and the VQC structure. It includes two required feature maps: `HRx` and 
`HRyRzCnot`.
* **VQC Training:** The scripts `img_vqc.py`, `text_vqc.py`, and `multimodal_vqc.py` perform training with four specific configurations:

| Configuration | Feature Map | Ansatz |
| :--- | :--- | :--- |
| **Config A** | HRx | MPS_TTN |
| **Config B** | HRx | TensorRing |
| **Config C** | HRyRzCnot | MPS_TTN |
| **Config D** | HRyRzCnot | TensorRing |

* **Baseline Testing:** `src/quantum/baseline.py` performs VQC training with the same configurations on provided input datasets. This was utilized for the **KDEF dataset**, located in `data/processed/baseline_KDEF`.
* **Results:** Model weights, performance plots, and metrics are stored in the `results/` folder.