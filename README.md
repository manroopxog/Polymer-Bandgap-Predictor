from google.colab import files

# 1. PASTE YOUR STREAMLIT APP LINK HERE inside the quotes:
app_link = "YOUR_LINK_HERE"

# 2. The script writes the README for you
readme_content = f"""# 🔬 μ-TEG Polymer Bandgap Screener (Model B)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)](https://streamlit.io/)

**Live Application:** [👉 Click here to launch the web app]({app_link})

## Project Overview
This repository contains "Model B," an interactive, high-throughput machine learning screening pipeline developed to discover and evaluate organic conductive polymers for **micro-Thermoelectric Generator (μ-TEG)** applications. 

Developed as the core computational methodology for a MEXT research proposal, the tool leverages a **Graph Attention Network (GAT)** to predict the electronic bandgap (E_g) of organic molecules directly from their 2D molecular topology (SMILES strings). By prioritizing narrow-bandgap, highly π-conjugated systems, this model drastically accelerates the initial screening phase of thermoelectric material discovery prior to physical synthesis.

## Core Features
* **Single Molecule Inference:** Instantaneous bandgap predictions with automated categorization (Excellent, Moderate, Insulator).
* **High-Throughput Batch Screening:** Upload a CSV of candidate SMILES strings to evaluate entire chemical libraries simultaneously.
* **Active In-Browser Fine-Tuning:** A built-in training loop allowing researchers to upload novel ground-truth datasets to actively retrain the model weights, correct biases, and download the updated `.pth` file directly from the web interface.

## Computational Methodology

### 1. Featurization (RDKit)
Molecular graphs are constructed using RDKit, extracting strictly 2D topological features to avoid the computational overhead of 3D conformer generation. Node features include:
* Atomic Number (C, N, O, S, F)
* Hybridization State (SP, SP2, SP3)
* Aromaticity
* Formal Charge

### 2. Network Architecture
The model utilizes a custom PyTorch Geometric GAT architecture:
* **3 Attention Layers:** 4-head, 4-head, and 6-head configurations.
* **Hidden Channels:** 128
* **Pooling:** Global Mean Pooling for graph-level regression.

### 3. Training Pipeline
The network weights were established through a rigorous multi-phase transfer learning approach:
* **Phase 1 (Chemical Intuition):** Trained on 5,000 diverse small molecules from the **QM9 Dataset** to establish baseline topological mapping to quantum properties.
* **Phase 2 (Thermoelectric Calibration):** Fine-tuned on 233 highly conjugated polymers from the **Harvard CEP Dataset**. This shifted the model's focus from wide-bandgap insulators to the narrow-bandgap regime required for thermoelectrics.
* **Phase 3 (Anchor Correction):** "Shock-calibrated" to correct dataset mode-collapse, ensuring the model accurately maps the relationship between extended π-conjugation length and electron mobility (e.g., successfully scaling bandgaps inversely with thiophene chain length).

## Known Scientific Limitations
* **The "Acene Bias":** Due to heavy anchor-weighting on Pentacene during Phase 3 bias correction, the current model weights exhibit an overfit towards pure acene structures (e.g., Benzene, Anthracene, and Pentacene may output identical threshold values). Thiophene-based polymer chains remain highly accurate. Future fine-tuning iterations will aim to decouple this specific aromatic shortcut.

## Local Installation

To run this application locally:

1. Clone the repository:
   ```bash
   git clone [https://github.com/ManroopXog/polymer-bandgap-predictor.git](https://github.com/ManroopXog/polymer-bandgap-predictor.git)
   cd polymer-bandgap-predictor
   
