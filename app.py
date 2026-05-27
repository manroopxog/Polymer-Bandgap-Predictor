import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
import numpy as np
import pandas as pd
import io
import py3Dmol
from stmol import showmol


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
    model.load_state_dict(torch.load('model_B_TEG_finetuned.pth', map_location=torch.device('cpu')))
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

# ==========================================
# TAB 1: PREDICTION & DISCOVERY STUDIO
# ==========================================
with tab1:
    import urllib.parse
    import requests
    from rdkit.Chem import AllChem

    st.markdown("Enter a SMILES string, or use the Mutator buttons to alter the chemistry.")
    
    if "bg_smiles_input" not in st.session_state:
        st.session_state.bg_smiles_input = "N#CC(C#N)=C1C=CC(=C(C#N)C#N)C=C1"

    def run_mutation(rxn_smarts):
        try:
            mol = Chem.MolFromSmiles(st.session_state.bg_smiles_input)
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            products = rxn.RunReactants((mol,))
            if products:
                new_mol = products[0][0] 
                Chem.SanitizeMol(new_mol)
                st.session_state.bg_smiles_input = Chem.MolToSmiles(new_mol)
        except:
            pass 

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧬 Mutate: Add Fluorine (-F)"):
            run_mutation('[cH:1]>>[c:1](F)')
    with col2:
        if st.button("🧬 Mutate: Add Cyano (-C#N)"):
            run_mutation('[cH:1]>>[c:1](C#N)')

    user_smiles = st.text_input("Current Molecule SMILES:", st.session_state.bg_smiles_input)
    
    if user_smiles != st.session_state.bg_smiles_input:
        st.session_state.bg_smiles_input = user_smiles

    if user_smiles:
        mol = Chem.MolFromSmiles(user_smiles)
        if mol is not None:
            st.subheader("Interactive 3D Geometry:")
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mblock = Chem.MolToMolBlock(mol)
            viewer = py3Dmol.view(width=400, height=400)
            viewer.addModel(mblock, "mol")
            viewer.setStyle({'stick': {}, 'sphere': {'radius': 0.4}})
            viewer.zoomTo()
            showmol(viewer, height=400, width=400)
        else:
            st.error("Invalid SMILES string. Please check your input.")

    if st.button("Predict Bandgap & Search PubChem", type="primary"):
        torch.manual_seed(42)
        model.eval() 
        
        with st.spinner("Calculating quantum features & pinging global databases..."):
            if scaler is None:
                st.error("Scaler file not found. Please ensure your bandgap scaler is uploaded.")
            else:
                # Use Model B's specific graph converter
                graph = smiles_to_graph(user_smiles)
                
                if graph is None:
                    st.error("Invalid SMILES string.")
                else:
                    batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
                    
                    with torch.no_grad():
                        # Model B only takes x, edge_index, and batch (no edge_attr)
                        scaled_pred = model(graph.x, graph.edge_index, batch).numpy()
                        predicted_ev = scaler.inverse_transform(scaled_pred)[0][0]
                    
                    st.subheader("🤖 AI Prediction Result")
                    st.metric(label="Predicted Bandgap", value=f"{predicted_ev:.3f} eV")
                    
                    if 1.0 <= predicted_ev <= 2.5:
                        st.success("✅ IDEAL BANDGAP (Optimal Semiconductor).")
                    elif 0.5 <= predicted_ev < 1.0:
                        st.warning("⚠️ NARROW BANDGAP (Approaching Metallic).")
                    elif 2.5 < predicted_ev <= 3.5:
                        st.warning("⚠️ WIDE BANDGAP (Approaching Insulator).")
                    else:
                        st.error("❌ INSULATOR / METALLIC (Outside viable semiconductor range).")

                    # PubChem API Logic
                    st.subheader("🌐 PubChem Reality Check")
                    try:
                        safe_smiles = urllib.parse.quote(user_smiles)
                        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{safe_smiles}/property/IUPACName,MolecularWeight,MolecularFormula,XLogP/JSON"
                        response = requests.get(url)
                        
                        if response.status_code == 200:
                            data = response.json()
                            props = data['PropertyTable']['Properties'][0]
                            
                            cid = props.get('CID', 'Unknown')
                            name = props.get('IUPACName', 'Complex Derivative (No standard name available)')
                            weight = props.get('MolecularWeight', 'Unknown')
                            formula = props.get('MolecularFormula', 'Unknown')
                            xlogp = props.get('XLogP', 'Data not available')
                            
                            st.info(f"**✅ Molecule Recognized (PubChem CID: {cid})**\n\n"
                                    f"**Formula:** {formula}\n\n"
                                    f"**IUPAC Name:** {name}\n\n"
                                    f"**Mass:** {weight} g/mol\n\n"
                                    f"**XLogP (Toxicity/Bioaccumulation proxy):** {xlogp}")
                            
                        elif response.status_code == 404 or response.status_code == 400:
                            st.success("🌟 **Novel Molecule!** No matches found in the PubChem database.")
                        else:
                            st.warning("Could not connect to PubChem API.")
                    except Exception as e:
                        st.warning(f"API Error: {e}")
                    
                    
# --- TAB 2: BATCH SCREENING ---
with tab2:
    st.subheader("CSV Batch Screening")
    st.markdown("Upload a CSV containing a list of candidate SMILES strings to evaluate them simultaneously.")
    
    # NEW: Explicit instructions for the CSV format
    st.info("📝 **CSV Format Required:** Your file must be a `.csv` containing at least one column with valid **SMILES strings**. You can have other columns (like molecule names or IDs); the model will ignore them and only process the SMILES column you select below.")
    
    uploaded_file = st.file_uploader("Upload your candidate CSV", type=["csv"], key="batch_uploader")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded dataset:", df.head(3))
        
        # The app lets you dynamically select which column has the SMILES
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
    
    # NEW: Explicit instructions for the training CSV format
    st.info("📝 **Training CSV Format Required:** Your file must contain at least two columns:\n"
            "1. **SMILES:** The molecular structure string.\n"
            "2. **Bandgap Target:** The known, true bandgap value in electron-volts (eV).\n"
            "*(Make sure there are no blank/empty cells in these columns, or those rows will be skipped).*")
    
    train_file = st.file_uploader("Upload Training Dataset (CSV)", type=["csv"], key="train_uploader")
    
    if train_file is not None:
        train_df = pd.read_csv(train_file)
        st.write("Dataset Preview:", train_df.head(3))
        
        col1, col2 = st.columns(2)
        # The app lets you map the required data to whatever your column names actually are
        train_smiles_col = col1.selectbox("Map your SMILES Column:", train_df.columns, index=0)
        train_target_col = col2.selectbox("Map your Bandgap Target Column (eV):", train_df.columns, index=1 if len(train_df.columns)>1 else 0)
        
        col3, col4, col5 = st.columns(3)
        epochs = col3.number_input("Epochs", min_value=1, max_value=500, value=50, step=10)
        lr = col4.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.0005, format="%.4f")
        batch_size = col5.number_input("Batch Size", min_value=4, max_value=64, value=16, step=4)
        
        if st.button("Start Fine-Tuning", type="primary"):
            # Clean data and drop NaNs
            train_df[train_target_col] = pd.to_numeric(train_df[train_target_col], errors='coerce')
            train_df = train_df.dropna(subset=[train_smiles_col, train_target_col])
            
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
                
                if train_graphs:
                    scaled_targets = scaler.transform(np.array(raw_targets).reshape(-1, 1))
                    for i, graph in enumerate(train_graphs):
                        graph.y = torch.tensor([scaled_targets[i]], dtype=torch.float)
            
            if len(train_graphs) == 0:
                st.error("No valid molecules found to train on.")
            else:
                loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = torch.nn.MSELoss()
                
                st.info(f"Training on {len(train_graphs)} molecules...")
                progress_bar = st.progress(0, text="Starting training...")
                loss_text = st.empty()
                
                model.train()
                
                for epoch in range(1, epochs + 1):
                    total_loss = 0
                    for data in loader:
                        optimizer.zero_grad()
                        out = model(data.x, data.edge_index, data.batch)
                        loss = criterion(out, data.y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        total_loss += loss.item()
                    
                    avg_loss = total_loss / len(loader)
                    progress_bar.progress(epoch / epochs, text=f"Epoch {epoch}/{epochs}")
                    loss_text.text(f"Current MSE Loss: {avg_loss:.4f}")
                
                st.success("Fine-tuning complete! The model weights have been updated in-memory.")
                
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
                
        
