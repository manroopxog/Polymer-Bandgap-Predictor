<div align="center">
  
<h1>μ-TEG Polymer Bandgap Screener</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg?logo=streamlit&logoColor=white" alt="Streamlit">
</p>

<h3>High-Throughput Deep Learning Pipeline for Organic Thermoelectric Discovery</h3>

<br>

<h2><a href="INSERT_YOUR_STREAMLIT_LINK_HERE">Access Live Application</a></h2>

<br>
</div>

<hr>

<h2>Project Overview</h2>
<p>
  This repository provides an integrated computational framework for the rapid screening of conductive organic polymers targeted at micro-Thermoelectric Generator (&mu;-TEG) applications. The objective of this project is to address the high computational cost of traditional density functional theory (DFT) methods by utilizing a Graph Attention Network (GAT) to predict the electronic bandgap (E<sub>g</sub>) of molecular structures directly from their 2D topological data.
</p>
<p>
  The pipeline is designed for scalability, allowing for the evaluation of chemical libraries through automated processing of SMILES strings, thereby facilitating the identification of narrow-bandgap candidates that exhibit superior electron mobility.
</p>

<h2>Computational Methodology</h2>
<p>
  The model architecture is built upon a PyTorch Geometric implementation of a Graph Attention Network (GAT). This approach is preferred for its ability to learn nuanced representations of molecular graphs by assigning varying levels of importance to neighbor nodes and edges, effectively capturing the long-range conjugation effects critical to semiconductor behavior.
</p>
<ul>
  <li><b>Topological Featurization:</b> Molecular structures are processed via RDKit to generate node-level features, including atomic number, hybridization state (sp/sp2/sp3), aromaticity, and formal charge.</li>
  <li><b>Feature Encoding:</b> The network maps these local features into a 128-dimensional hidden space, allowing for complex non-linear correlations between molecular geometry and electronic bandgaps.</li>
  <li><b>Global Pooling:</b> The architecture utilizes global mean pooling to aggregate node embeddings into a single graph-level vector, which is subsequently mapped to a continuous regression output representing the bandgap in electron-volts (eV).</li>
</ul>

<h2>Training Pipeline</h2>
<p>
  The model employs a multi-phase transfer learning strategy to optimize predictive accuracy:
</p>
<ol>
  <li><b>Phase I: Generalization:</b> Baseline training was conducted on the QM9 dataset (5,000 molecules) to develop a robust understanding of fundamental molecular topology.</li>
  <li><b>Phase II: Domain Adaptation:</b> The weights were fine-tuned using the Harvard Clean Energy Project (CEP) dataset, specifically targeting 233 highly conjugated polymers to calibrate the model for narrow-bandgap phenomena.</li>
  <li><b>Phase III: Calibration:</b> An anchor-based fine-tuning protocol was implemented to resolve mode-collapse and ensure the model accurately replicates the inverse relationship between $\pi$-conjugation length and bandgap energy.</li>
</ol>

<h2>System Functionality</h2>
<ul>
  <li><b>Single Molecule Inference:</b> An interactive interface for real-time inference on individual molecular structures.</li>
  <li><b>Batch Processing:</b> A dedicated utility for evaluating large-scale CSV-based chemical libraries, automating the extraction of bandgap predictions for thousands of candidates.</li>
  <li><b>Dynamic Fine-Tuning:</b> An in-browser training module that enables users to input proprietary ground-truth datasets, modify learning parameters, and export optimized model weights (via .pth files) to adapt the network to novel chemical classes.</li>
</ul>

<h2>Limitations and Model Bias</h2>
<p>
  Users should be aware of the "Acene Bias" currently present in the model weights. During the Phase III calibration, the model was heavily weighted toward acene-based aromatics (such as pentacene and anthracene) to force the regression into a conductive regime. As a result, the model may output highly similar bandgap values for structurally related acenes regardless of subtle topological variations. This bias is acknowledged as a constraint of the current training set, and the active fine-tuning module is provided to allow for the correction of these trends in future iterations.
</p>

<hr>

<h2>Local Installation and Dependencies</h2>
<p>To run this repository locally, ensure Python 3.10 is installed. Dependencies can be resolved via the provided configuration:</p>

<pre><code>git clone https://github.com/ManroopXog/polymer-bandgap-predictor.git
cd polymer-bandgap-predictor
pip install -r requirements.txt
streamlit run app.py</code></pre>

<br>

<p>
  <b>Author:</b> Manroop Manota | B.Sc. Chemistry Honours, Swami Shraddhanand College, University of Delhi
</p>
