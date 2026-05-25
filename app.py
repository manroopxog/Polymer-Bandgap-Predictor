import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
import joblib
import os
import io

# ==========================================
# 1. PAGE SETUP & UI
# ==========================================
st.set_page_config(page_title="Model B: Bandgap Screener", layout="wide")
st.title("⚡ Model B: Polymer Bandgap Predictor")

# ==========================================
# 2. DEFINE THE NEURAL ARCHITECTURE
# ==========================================
class BandgapPredictorGAT(torch.nn.Module):
    def __init__(self, num_node_features):
        super(BandgapPredictorGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 64, heads=2, edge_dim=1)
        self.conv2 = GATConv(128, 64, heads=2, edge_dim=1)
        self.conv3 = GATConv(128, 64, heads=2, edge_dim=1)
        self.conv4 = GATConv(128, 64, heads=1, edge_dim=1)
        self.dropout = Dropout(p=0.2) 
        self.linear1 = Linear(64, 32)
        self.linear2 = Linear(32, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = edge_attr.view(-1, 1)
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv4(x, edge_index, edge_attr=edge_attr))
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def get_unified_features(atom):
    return [
        float(atom.GetAtomicNum()), float(atom.GetChiralTag()), float(atom.GetDegree()), 
        float(atom.GetFormalCharge()), float(atom.GetHybridization()), float(atom.GetIsAromatic()), 
        float(atom.GetMass()), float(atom.GetNumImplicitHs()), float(atom.GetNumRadicalElectrons()), 
        float(atom.GetTotalValence()), float(atom.GetTotalNumHs())
    ]

# Helper function to process SMILES into PyTorch Geometric Graphs
def process_smiles_to_graphs(df, target_col=None, scaler=None, is_training=False):
    graphs = []
    valid_indices = []
    
    for i, row in df.iterrows():
        smiles = str(row['smiles'])
        mol = Chem.MolFromSmiles(smiles)
        
        if not mol: continue
        mol = Chem.AddHs(mol)
        
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except: continue
        
        x = torch.tensor([get_unified_features(a) for a in mol.GetAtoms()], dtype=torch.float)
        edges, dists = [], []
        conf = mol.GetConformer()
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            d = conf.GetAtomPosition(a1).Distance(conf.GetAtomPosition(a2))
            edges.extend([[a1, a2], [a2, a1]])
            dists.extend([d, d])
            
        if not edges: continue
            
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(dists, dtype=torch.float)
        
        if is_training and target_col:
            y_val = row[target_col]
            y_scaled = scaler.transform([[y_val]])[0][0]
            y_tensor = torch.tensor([[y_scaled]], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
        else:
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
        graphs.append(data)
        valid_indices.append(i)
        
    return graphs, valid_indices

# ==========================================
# 3. LOAD THE PRE-TRAINED BRAIN (CACHED)
# ==========================================
@st.cache_resource
def load_model_and_scaler():
    device = torch.device('cpu') 
    model = BandgapPredictorGAT(num_node_features=11).to(device)
    
    model_path = 'gat_custom_finetuned (1).pth'
    scaler_path = 'gat_harvard_scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        scaler = joblib.load(scaler_path)
        model.eval()
        return model, scaler, True
    else:
        return model, None, False

model, scaler, is_loaded = load_model_and_scaler()

if not is_loaded:
    st.warning("⚠️ Pre-trained weights not found! Please upload 'gat_harvard_finetuned (1).pth' and 'gat_harvard_scaler.pkl' to the repository. Predictions will be disabled until weights are loaded.")

# ==========================================
# 4. APP TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["🧪 Single Molecule Test", "⚡ Batch Screening", "🔧 Fine-Tune Model"])

# ------------------------------------------
# TAB 1: SINGLE MOLECULE TEST
# ------------------------------------------
with tab1:
    st.subheader("Quick Bandgap Prediction")
    st.markdown("Paste a single SMILES string to instantly predict its HOMO-LUMO bandgap.")
    
    single_smiles = st.text_input("Enter SMILES string:", placeholder="e.g., N#CC(C#N)=C1C=CC(=C(C#N)C#N)C=C1")
    
    if st.button("Predict Single Molecule"):
        if not is_loaded:
            st.error("Cannot predict: Model weights are missing.")
        elif not single_smiles:
            st.warning("Please enter a SMILES string.")
        else:
            with st.spinner("Calculating 3D geometry and predicting..."):
                # Wrap the single SMILES in a temporary dataframe to use our helper function
                temp_df = pd.DataFrame({'smiles': [single_smiles]})
                graphs, valid_indices = process_smiles_to_graphs(temp_df, is_training=False)
                
                if len(graphs) == 0:
                    st.error("❌ Invalid SMILES or RDKit failed to generate 3D geometry.")
                else:
                    data = graphs[0]
                    batch = torch.zeros(data.x.size(0), dtype=torch.long)
                    with torch.no_grad():
                        scaled_pred = model(data.x, data.edge_index, data.edge_attr, batch).numpy()
                        bg = scaler.inverse_transform(scaled_pred)[0][0]
                        
                    st.success(f"**Predicted Bandgap:** {bg:.3f} eV")

# ------------------------------------------
# TAB 2: BATCH SCREENING
# ------------------------------------------
with tab2:
    st.subheader("Virtual Screening Pipeline")
    st.markdown("Upload a CSV file with a `smiles` column (e.g., the output from Model A).")
    
    uploaded_file = st.file_uploader("Upload CSV for Prediction", type=['csv'], key="predict_upload")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'smiles' not in df.columns:
            st.error("❌ CSV must contain a column named 'smiles'.")
        else:
            st.dataframe(df.head())
            if st.button("Predict Batch Bandgaps"):
                if not is_loaded:
                    st.error("Cannot predict: Model weights are missing.")
                else:
                    with st.spinner("Processing 3D molecular geometries..."):
                        graphs, valid_indices = process_smiles_to_graphs(df, is_training=False)
                    
                    if len(graphs) == 0:
                        st.error("No valid 3D graphs could be generated from the provided SMILES.")
                    else:
                        progress_bar = st.progress(0)
                        predictions = [None] * len(df)
                        
                        for idx, data in enumerate(graphs):
                            batch = torch.zeros(data.x.size(0), dtype=torch.long)
                            with torch.no_grad():
                                scaled_pred = model(data.x, data.edge_index, data.edge_attr, batch).numpy()
                                bg = scaler.inverse_transform(scaled_pred)[0][0]
                                predictions[valid_indices[idx]] = round(bg, 3)
                            progress_bar.progress((idx + 1) / len(graphs))
                            
                        df['Predicted_Bandgap'] = predictions
                        st.success("✅ Screening Complete!")
                        st.dataframe(df)
                        
                        csv_export = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results 📥",
                            data=csv_export,
                            file_name="Model_B_Results.csv",
                            mime="text/csv",
                        )

# ------------------------------------------
# TAB 3: FINE-TUNE MODEL
# ------------------------------------------
with tab3:
    st.subheader("Fine-Tune with Custom Molecules")
    st.markdown("Upload a CSV containing `smiles` and a target column (e.g., `bandgap`) to train the model further.")
    
    train_file = st.file_uploader("Upload CSV for Training", type=['csv'], key="train_upload")
    
    if train_file:
        train_df = pd.read_csv(train_file)
        st.dataframe(train_df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Select Target Column", train_df.columns)
            epochs = st.slider("Epochs", min_value=10, max_value=200, value=50, step=10)
        with col2:
            lr = st.selectbox("Learning Rate", [1e-3, 1e-4, 1e-5], index=2)
            batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
            
        if st.button("Start Training"):
            if 'smiles' not in train_df.columns:
                st.error("❌ CSV must contain a 'smiles' column.")
            else:
                train_df = train_df.dropna(subset=['smiles', target_col])
                
                # Fit a new scaler for the custom dataset
                new_scaler = StandardScaler()
                new_scaler.fit(train_df[[target_col]])
                
                with st.spinner("Building molecular graphs..."):
                    graphs, _ = process_smiles_to_graphs(train_df, target_col=target_col, scaler=new_scaler, is_training=True)
                
                if len(graphs) == 0:
                    st.error("Failed to generate graphs from the provided data.")
                else:
                    st.info(f"Training on {len(graphs)} molecules...")
                    device = torch.device('cpu')
                    
                    # Use existing loaded model if it exists to fine-tune
                    train_model = BandgapPredictorGAT(num_node_features=11).to(device)
                    if is_loaded:
                        train_model.load_state_dict(model.state_dict())
                    
                    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
                    optimizer = torch.optim.Adam(train_model.parameters(), lr=lr)
                    loss_fn = torch.nn.MSELoss()
                    
                    train_model.train()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for epoch in range(1, epochs + 1):
                        total_loss = 0
                        for data in loader:
                            optimizer.zero_grad()
                            out = train_model(data.x, data.edge_index, data.edge_attr, data.batch)
                            loss = loss_fn(out, data.y)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()
                        
                        progress_bar.progress(epoch / epochs)
                        status_text.text(f"Epoch {epoch}/{epochs} | Loss: {total_loss/len(loader):.4f}")
                        
                    st.success("✅ Training Complete!")
                    
                    # Save weights and scaler to memory buffers for downloading
                    buffer_model = io.BytesIO()
                    torch.save(train_model.state_dict(), buffer_model)
                    
                    buffer_scaler = io.BytesIO()
                    joblib.dump(new_scaler, buffer_scaler)
                    
                    st.markdown("### Download Updated Weights")
                    st.markdown("Replace the old `.pth` and `.pkl` files in your GitHub repository with these new ones to make the updates permanent.")
                    
                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        st.download_button(
                            label="Download fine_tuned.pth",
                            data=buffer_model.getvalue(),
                            file_name="gat_custom_finetuned.pth",
                            mime="application/octet-stream"
                        )
                    with d_col2:
                        st.download_button(
                            label="Download scaler.pkl",
                            data=buffer_scaler.getvalue(),
                            file_name="gat_custom_scaler.pkl",
                            mime="application/octet-stream"
                        )
