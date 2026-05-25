import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from rdkit import Chem
import joblib
import numpy as np
import pandas as pd

# ==========================================
# 1. DEFINE THE GAT MODEL ARCHITECTURE
# ==========================================
class GATModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=4, concat=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=6, concat=False)
        
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = F.elu(x)
        x = self.lin2(x)
        return x

# ==========================================
# 2. DEFINE THE FEATURIZER
# ==========================================
def get_node_features(atom):
    features = []
    atomic_num = atom.GetAtomicNum()
    features += [int(atomic_num == i) for i in [6, 7, 8, 16, 9]]
    hybridization = atom.GetHybridization()
    features += [
        int(hybridization == Chem.rdchem.HybridizationType.SP),
        int(hybridization == Chem.rdchem.HybridizationType.SP2),
        int(hybridization == Chem.rdchem.HybridizationType.SP3)
    ]
    features.append(int(atom.GetIsAromatic()))
    features.append(float(atom.GetFormalCharge()))
    return features

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    node_features = [get_node_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

# ==========================================
# 3. LOAD ASSETS
# ==========================================
st.set_page_config(page_title="μ-TEG Model B | Bandgap Screener", layout="wide")

@st.cache_resource
def load_assets():
    model = GATModel(num_node_features=10, hidden_channels=128)
    model.load_state_dict(torch.load('model_B_TEG.pth', map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load('harvard_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Failed to load model or scaler. Error: {e}")
    st.stop()

# ==========================================
# 4. STREAMLIT UI & TABS
# ==========================================
st.title("🔬 Model B: High-Throughput Bandgap Screener")
st.markdown("Evaluate conductive organic polymers for micro-Thermoelectric Generator (μ-TEG) applications.")

# Create navigation tabs
tab1, tab2, tab3 = st.tabs(["Single Molecule", "Batch Screening", "Model & Training Info"])

# --- TAB 1: SINGLE MOLECULE ---
with tab1:
    st.subheader("Single Molecule Prediction")
    smiles_input = st.text_input("Enter SMILES string:", placeholder="e.g., c1cc(sc1)c2ccsc2")

    if st.button("Predict Bandgap", key="single"):
        if not smiles_input:
            st.warning("Please enter a SMILES string.")
        else:
            with st.spinner("Analyzing graph topology..."):
                graph = smiles_to_graph(smiles_input)
                if graph is None:
                    st.error("Invalid SMILES string.")
                else:
                    batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
                    with torch.no_grad():
                        scaled_prediction = model(graph.x, graph.edge_index, batch).numpy()
                    real_bandgap = scaler.inverse_transform(scaled_prediction)[0][0]
                    
                    st.success("Analysis Complete!")
                    st.metric(label="Predicted Bandgap (E_g)", value=f"{real_bandgap:.4f} eV")
                    
                    if real_bandgap < 1.5:
                        st.info("💡 **Excellent Candidate!** Narrow bandgap suggests high electrical conductivity.")
                    elif real_bandgap < 3.0:
                        st.warning("⚠️ **Moderate Bandgap.** May require heavy doping.")
                    else:
                        st.error("🛑 **Wide Bandgap.** Likely an insulator.")

# --- TAB 2: BATCH SCREENING ---
with tab2:
    st.subheader("CSV Batch Screening")
    st.markdown("Upload a CSV containing a list of candidate SMILES strings to evaluate them simultaneously.")
    
    uploaded_file = st.file_uploader("Upload your candidate CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded dataset:", df.head(3))
        
        # Let user select which column contains the SMILES strings
        smiles_col = st.selectbox("Select the column containing SMILES:", df.columns)
        
        if st.button("Run Batch Screening", key="batch"):
            progress_text = "Processing molecules..."
            my_bar = st.progress(0, text=progress_text)
            
            predictions = []
            total = len(df)
            
            for i, smiles in enumerate(df[smiles_col]):
                try:
                    graph = smiles_to_graph(str(smiles))
                    if graph is not None:
                        batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
                        with torch.no_grad():
                            scaled_pred = model(graph.x, graph.edge_index, batch).numpy()
                        real_bg = scaler.inverse_transform(scaled_pred)[0][0]
                        predictions.append(round(real_bg, 4))
                    else:
                        predictions.append("Invalid SMILES")
                except Exception:
                    predictions.append("Error")
                
                # Update progress bar
                my_bar.progress((i + 1) / total, text=f"Processed {i+1}/{total} molecules")
            
            df['Predicted_Bandgap_eV'] = predictions
            st.success("Batch screening complete!")
            st.dataframe(df)
            
            # Download button for the results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Screening Results (CSV)",
                data=csv,
                file_name="teg_candidates_screened.csv",
                mime="text/csv",
            )

# --- TAB 3: TRAINING INFO ---
with tab3:
    st.subheader("Architecture & Training History")
    st.markdown("""
    **Model Architecture:** Graph Attention Network (GAT)
    * **Input Features:** 10 (Atomic Number, Hybridization, Aromaticity, Formal Charge)
    * **Attention Layers:** 3 Layers (4 heads, 4 heads, 6 heads)
    * **Hidden Channels:** 128
    
    **Phase 1 Training: Chemical Intuition**
    * **Dataset:** QM9 Subset (5,000 diverse molecules)
    * **Epochs:** 300
    * **Purpose:** Establish baseline 2D topological mapping to quantum properties.
    
    **Phase 2 Training: Transfer Learning & Bias Correction**
    * **Dataset:** Harvard CEP Subset (233 highly conjugated polymers)
    * **Epochs:** 300
    * **Purpose:** Shift model weights away from small molecule wide-bandgap bias to accurately predict narrow-bandgap phenomena required for micro-Thermoelectric Generators.
    """)
    
