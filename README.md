<div align="center">
  
<h1>🔬 μ-TEG Polymer Bandgap Screener</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg?logo=streamlit&logoColor=white" alt="Streamlit">
</p>

<h3>High-Throughput Deep Learning for Organic Thermoelectric Discovery</h3>

<br>

<h2><a href="INSERT_YOUR_STREAMLIT_LINK_HERE">🚀 Launch Live Application Here</a></h2>

<br>
</div>

<hr>

<h2>📖 Project Overview</h2>
<p>
  <b>Model B</b> is an interactive machine learning pipeline designed to accelerate the discovery of organic conductive polymers for <b>micro-Thermoelectric Generator (&mu;-TEG)</b> applications. 
</p>
<p>
  By leveraging a custom <b>Graph Attention Network (GAT)</b>, this tool predicts the electronic bandgap (E<sub>g</sub>) of molecules directly from their 2D topology (SMILES strings). It bypasses the computationally expensive 3D conformer generation process and is specifically calibrated to identify highly &pi;-conjugated, narrow-bandgap systems capable of high electron mobility.
</p>

<h2>✨ Key Features</h2>
<ul>
  <li><b>🔍 Single Molecule Inference:</b> Instantaneous bandgap predictions with automated material categorization.</li>
  <li><b>📚 High-Throughput Batch Screening:</b> Upload bulk <code>.csv</code> files to evaluate entire chemical libraries simultaneously.</li>
  <li><b>⚙️ Active In-Browser Fine-Tuning:</b> A built-in training module that allows researchers to upload novel ground-truth datasets, dynamically correct model bias, and download updated <code>.pth</code> weights directly from the web interface.</li>
</ul>

<h2>🧠 Computational Architecture</h2>
<p>The model utilizes a PyTorch Geometric GAT architecture trained through a multi-phase transfer learning approach:</p>
<ul>
  <li><b>Topological Featurization:</b> RDKit extraction of Atomic Number, Hybridization, Aromaticity, and Formal Charge.</li>
  <li><b>Phase 1 (Chemical Intuition):</b> Baseline training on 5,000 diverse molecules (QM9 subset).</li>
  <li><b>Phase 2 (Thermoelectric Calibration):</b> Fine-tuned on 233 highly conjugated polymers (Harvard CEP subset).</li>
</ul>
