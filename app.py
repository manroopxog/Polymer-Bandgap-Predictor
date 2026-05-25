import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from rdkit import Chem
import joblib
import numpy as np

# ==========================================
# 1. DEFINE THE EXACT GAT MODEL ARCHITECTURE
# ==========================================
class GATModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=4, concat=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=6, concat=False)
        
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1) # Predicts 1 value (Bandgap)

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
# 2. DEFINE THE BULLETPROOF FEATURIZER
# ==========================================
def get_node_features(atom):
    features = []
    atomic_num = atom.GetAtomicNum()
    features += [int(atomic_num == i) for i in [6, 7, 8, 16, 9]] # C, N, O, S, F
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
    if mol is None:
        return None
    
    node_features = [get_node_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)
    
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
        
    # Handle single-atom edge cases gracefully
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

# ==========================================
# 3. STREAMLIT UI & INFERENCE LOGIC
# ==========================================
st.set_page_config(page_title="μ-TEG Model B | Bandgap Screener", layout="centered")

st.title("🔬 Model B: Bandgap Screener")
st.markdown("Evaluate organic polymers for micro-Thermoelectric Generator (μ-TEG) applications. Input a candidate SMILES string to predict its electronic bandgap.")

# Load the Model and Scaler using Streamlit's cache so it only happens once
@st.cache_resource
def load_assets():
    # 10 features per node, 128 hidden channels (from our QM9/Harvard training)
    model = GATModel(num_node_features=10, hidden_channels=128)
    model.load_state_dict(torch.load('model_B_TEG.pth', map_location=torch.device('cpu')))
    model.eval() # Set to evaluation mode!
    
    scaler = joblib.load('harvard_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Failed to load model or scaler files. Ensure they are in the directory. Error: {e}")
    st.stop()

# User Input
smiles_input = st.text_input("Enter SMILES string:", placeholder="e.g., c1cc(sc1)c2ccsc2")

if st.button("Predict Bandgap"):
    if not smiles_input:
        st.warning("Please enter a SMILES string.")
    else:
        with st.spinner("Analyzing molecular graph..."):
            graph = smiles_to_graph(smiles_input)
            
            if graph is None:
                st.error("Invalid SMILES string. RDKit could not parse the molecule.")
            else:
                # Create a batch vector of zeros (since we are only predicting 1 molecule)
                batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
                
                # Run inference without tracking gradients
                with torch.no_grad():
                    scaled_prediction = model(graph.x, graph.edge_index, batch).numpy()
                
                # CRITICAL STEP: Reverse the scaling to get the true eV value
                real_bandgap = scaler.inverse_transform(scaled_prediction)[0][0]
                
                # Display Results
                st.success("Analysis Complete!")
                st.metric(label="Predicted Bandgap (E_g)", value=f"{real_bandgap:.4f} eV")
                
                if real_bandgap < 1.5:
                    st.info("💡 **Excellent Candidate!** This narrow bandgap suggests high potential for electrical conductivity in a thermoelectric generator.")
                elif real_bandgap < 3.0:
                    st.warning("⚠️ **Moderate Bandgap.** May require heavy doping to achieve necessary conductivity.")
                else:
                    st.error("🛑 **Wide Bandgap.** Likely an insulator. Poor candidate for μ-TEGs.")
