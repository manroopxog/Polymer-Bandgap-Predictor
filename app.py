import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
import joblib
import numpy as np
import pandas as pd
import io

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

def smiles_to_graph(smiles, bandgap=None):
    mol = Chem.MolFromSmiles(str(smiles))
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
    
    # If bandgap is provided (for training), attach it as the target (y)
    if bandgap is not None:
        y = torch.tensor([[bandgap]], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y)
    
    return Data(x=x, edge_index=edge_index)

# ==========================================
# 3. LOAD ASSETS
# ==========================================
st.set_page_config(page_title="μ-TEG Model B | Bandgap Screener", layout="wide")

@st.cache_resource
def load_assets():
    model = GATModel(num_node_features=10, hidden_channels=128)
    model.load_state_dict(torch.load('model_B_TEG.pth', map_location=torch.device('cpu')))
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
tab1, tab2, tab3 = st.tabs(["Single Molecule", "Batch Screening", "Active Fine-Tuning"])

# --- TAB 1: SINGLE MOLECULE ---
with tab1:
    st.subheader("Single Molecule Prediction")
    smiles_input = st.text_input("Enter SMILES string:", placeholder="e.g., c1cc(sc1)c2ccsc2")

    if st.button("Predict Bandgap", key="single"):
        if not smiles_input:
            st.warning("Please enter a SMILES string.")
        else:
            with st.spinner("Analyzing graph topology..."):
                model.eval() # Ensure evaluation mode
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
    uploaded_file = st.file_uploader("Upload your candidate CSV", type=["csv"], key="batch_uploader")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded dataset:", df.head(3))
        smiles_col = st.selectbox("Select the column containing SMILES:", df.columns, key="batch_smiles")
        
        if st.button("Run Batch Screening", key="batch_run"):
            progress_text = "Processing molecules..."
            my_bar = st.progress(0, text=progress_text)
            
            predictions = []
            total = len(df)
            model.eval()
            
            for i, smiles in enumerate(df[smiles_col]):
                try:
                    graph = smiles_to_graph(smiles)
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
                my_bar.progress((i + 1) / total, text=f"Processed {i+1}/{total} molecules")
            
            df['Predicted_Bandgap_eV'] = predictions
            st.success("Batch screening complete!")
            st.dataframe(df)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Screening Results (CSV)", data=csv, file_name="teg_candidates_screened.csv", mime="text/csv")

# --- TAB 3: ACTIVE FINE-TUNING ---
with tab3:
    st.subheader("Retrain & Fine-Tune Model Weights")
    st.markdown("Upload a CSV containing ground-truth bandgap values to further adapt the Graph Attention Network to new polymers.")
    
    train_file = st.file_uploader("Upload Training Dataset (CSV)", type=["csv"], key="train_uploader")
    
    if train_file is not None:
        train_df = pd.read_csv(train_file)
        st.write("Dataset Preview:", train_df.head(3))
        
        col1, col2 = st.columns(2)
        train_smiles_col = col1.selectbox("SMILES Column:", train_df.columns, index=0)
        train_target_col = col2.selectbox("Bandgap Target Column (eV):", train_df.columns, index=1 if len(train_df.columns)>1 else 0)
        
        col3, col4, col5 = st.columns(3)
        epochs = col3.number_input("Epochs", min_value=1, max_value=500, value=50, step=10)
        lr = col4.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.0005, format="%.4f")
        batch_size = col5.number_input("Batch Size", min_value=4, max_value=64, value=16, step=4)
        
        if st.button("Start Fine-Tuning", type="primary"):
            # 1. Clean data and drop NaNs
            train_df[train_target_col] = pd.to_numeric(train_df[train_target_col], errors='coerce')
            train_df = train_df.dropna(subset=[train_smiles_col, train_target_col])
            
            # 2. Process Graphs
            with st.spinner("Converting SMILES to Graph representations..."):
                train_graphs = []
                raw_targets = []
                for _, row in train_df.iterrows():
                    smiles = row[train_smiles_col]
                    target = row[train_target_col]
                    graph = smiles_to_graph(smiles, bandgap=target)
                    if graph is not None:
                        train_graphs.append(graph)
                        raw_targets.append(target)
                
                # 3. Apply the existing scaler to the new targets
                if train_graphs:
                    scaled_targets = scaler.transform(np.array(raw_targets).reshape(-1, 1))
                    for i, graph in enumerate(train_graphs):
                        graph.y = torch.tensor([scaled_targets[i]], dtype=torch.float)
            
            if len(train_graphs) == 0:
                st.error("No valid molecules found to train on.")
            else:
                # 4. Training Loop setup
                loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = torch.nn.MSELoss()
                
                st.info(f"Training on {len(train_graphs)} molecules...")
                progress_bar = st.progress(0, text="Starting training...")
                loss_text = st.empty()
                
                model.train() # Set to training mode!
                
                for epoch in range(1, epochs + 1):
                    total_loss = 0
                    for data in loader:
                        optimizer.zero_grad()
                        out = model(data.x, data.edge_index, data.batch)
                        loss = criterion(out, data.y)
                        loss.backward()
                        # Crucial: Gradient clipping prevents NaNs during live retraining
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        total_loss += loss.item()
                    
                    avg_loss = total_loss / len(loader)
                    progress_bar.progress(epoch / epochs, text=f"Epoch {epoch}/{epochs}")
                    loss_text.text(f"Current MSE Loss: {avg_loss:.4f}")
                
                st.success("Fine-tuning complete! The model weights have been updated in-memory.")
                
                # 5. Provide Download for the New Model
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                
                st.download_button(
                    label="💾 Download Updated Model Weights (.pth)",
                    data=buffer,
                    file_name="model_B_TEG_finetuned.pth",
                    mime="application/octet-stream",
                    help="Replace your old model_B_TEG.pth file with this one to keep these new improvements permanently."
                )
