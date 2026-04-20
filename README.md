# NetAlign_gml: Network Alignment using Node2Vec

## 📌 Project Overview
This project implements a **network alignment pipeline** that matches nodes across two graphs using representation learning.
The key idea is:
> Learn node embeddings independently for two graphs, then align nodes based on embedding similarity.
---

## 🧠 Method
The pipeline consists of three main components:
### 1. Node Embedding (Node2Vec)
Each node is mapped into a vector space:
\[
f: V \rightarrow \mathbb{R}^d
\]
We use **Node2Vec** to capture structural similarity through random walks.

---

### 2. Similarity Computation
We compute cosine similarity between node embeddings:

\[
\text{sim}(u, v) = \frac{f(u) \cdot f(v)}{\|f(u)\| \|f(v)\|}
\]

---

### 3. Node Alignment
For each node in Graph 1, we find the most similar node in Graph 2.

---

## 📂 Project Structure
NetAlign_gml/
│
├── data/
│ ├── raw/ # original graphs
│ └── processed/ # processed data
│
├── results/
│ └── experiment_results.csv # alignment output
│
├── src/ # core functions (embedding, alignment)
│
├── NetAlign.ipynb # main pipeline
├── .gitignore
└── README.md

---

## ⚙️ Installation
```bash
git clone https://github.com/Constance930786/NetAlign_gml.git
cd NetAlign_gml

Create environment:

conda create -n netalign python=3.9
conda activate netalign

Install dependencies:
pip install networkx numpy pandas scikit-learn gensim
📊 Dataset
Place your graph files into:

data/raw/
Supported formats:

.edgelist
.txt
.csv

Example:

data/raw/graph1.edgelist
data/raw/graph2.edgelist
🚀 How to Run
Step 1: Open notebook
jupyter notebook NetAlign.ipynb
Step 2: Run pipeline

The notebook includes the full pipeline:

Load graph data
Preprocess graphs
Generate embeddings using Node2Vec
Compute similarity matrix
Perform node alignment
Save results
📈 Output

Results are saved to:

results/experiment_results.csv

Example:
node_G1	node_G2	similarity
0	5	0.92

🔬 ## Pipeline Overview
Authentic Twitter Single Image
    ↓
Read base graph G
    ↓
Construct Node Features X
    ↓
permutation
    ↓
Generate (Gs, Gt, Xs, Xt, gt_map)
    ↓
Add Structural Noise / Attribute Noise
    ↓
Running the Alignment Method:
    - Node2Vec + NN
    - FINAL-style
    - WAlign-inspired
    ↓
Evaluate:
    - Accuracy@1
    - Hit@1
    - Hit@5
    ↓
Save CSV
    ↓
Draw robustness graph
    ↓
draw the conclusion

🧪 Evaluation (Optional Extension)
You can evaluate alignment performance using:

Accuracy
Accuracy
=
correct matches
𝑁
Accuracy=N
correct matches​
Hits@K

Check whether the true match appears in top-K candidates.

📌 Key Insight
Node2Vec captures graph structure
Similar nodes across graphs have similar embeddings
Alignment becomes a nearest-neighbor problem

🧪 Future Work
Hungarian Algorithm for global alignment
Graph Neural Networks (GNN)
Cross-domain network alignment
Semi-supervised alignment with anchor nodes

📚 References
Grover & Leskovec, Node2Vec: Scalable Feature Learning for Networks

👩‍💻 Author
Mengzi Cheng
University of Virginia

⭐ Notes
data/ may be empty in GitHub (ignored files)
results/ contains generated outputs
Run notebook to reproduce results
