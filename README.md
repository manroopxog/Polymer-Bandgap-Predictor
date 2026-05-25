# μ-TEG Polymer Bandgap Screener (Model B)

**Live Application:** [👉 Click here to open the Live App](INSERT_YOUR_STREAMLIT_LINK_HERE)

## Overview
This is an interactive machine learning web app designed to screen organic conductive polymers for micro-Thermoelectric Generator (μ-TEG) applications. It uses a custom Graph Attention Network (GAT) to predict the electronic bandgap of molecules directly from their SMILES strings.

## Key Features
* **Single Molecule Prediction:** Check the bandgap of individual SMILES strings instantly.
* **Batch Screening:** Upload a CSV file to evaluate multiple candidates at once.
* **Active Fine-Tuning:** Upload your own dataset to retrain the model directly in the browser.

## How to Run Locally
If you want to run this project on your own machine instead of the cloud:

1. Clone this repository.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Launch the app: `streamlit run app.py`
