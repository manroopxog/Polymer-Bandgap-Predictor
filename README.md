<div align="center">
  <h1>🔬 &mu;-TEG Polymer Bandgap Screener</h1>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.10-blue.svg?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg?logo=streamlit&logoColor=white" alt="Streamlit">
  </p>

  <h3>High-Throughput Deep Learning for Organic Thermoelectric Discovery</h3>
  
  <h2><a href="INSERT_YOUR_STREAMLIT_LINK_HERE">🚀 Launch Live Application Here</a></h2>
</div>

<hr>

<h2>📖 Project Overview</h2>
<p><b>Model B</b> is an interactive machine learning pipeline designed to accelerate the discovery of organic conductive polymers for <b>micro-Thermoelectric Generator (&mu;-TEG)</b> applications.</p>
<p>By leveraging a custom Graph Attention Network (GAT), this tool predicts the electronic bandgap (E<sub>g</sub>) of molecules directly from their 2D topology (SMILES strings). It is specifically calibrated to identify highly &pi;-conjugated, narrow-bandgap systems.</p>

<h2>✨ Core Functionality</h2>
<ul>
  <li><b>🔍 Single Molecule Inference:</b> Instantaneous bandgap predictions.</li>
  <li><b>📚 High-Throughput Batch Screening:</b> Upload <code>.csv</code> files to evaluate chemical libraries.</li>
  <li><b>⚙️ Active In-Browser Fine-Tuning:</b> Retrain model weights and download the updated <code>.pth</code> file directly.</li>
</ul>

<h2>💻 Local Installation</h2>
<pre><code>git clone https://github.com/ManroopXog/polymer-bandgap-predictor.git
cd polymer-bandgap-predictor
pip install -r requirements.txt
streamlit run app.py</code></pre>

<p><b>Author:</b> Manroop Manota | B.Sc. Chemistry Honours, Swami Shraddhanand College, University of Delhi</p>
