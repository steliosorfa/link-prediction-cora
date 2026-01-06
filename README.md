# Link Prediction on Cora Graph

Course: Τεχνολογίες Γραφημάτων  
Department: Πληροφορικής και Τηλεματικής  
University: Χαροκόπειο Πανεπιστήμιο  

## Description
This project studies the problem of link prediction on graphs using three different approaches:

- Heuristic-based methods (Common Neighbors, Jaccard, Adamic–Adar)
- Shallow node embeddings (Node2Vec + MLP)
- End-to-End Graph Neural Networks (GCN)

The Cora citation network is used as a benchmark dataset.

## Requirements
Install dependencies using:

pip install -r requirements.txt

## Execution
Run the main script using:

python main.py

The script:

prepares and preprocesses the Cora dataset,

trains and evaluates all link prediction methods,

reports AUC scores,

produces visualizations (t-SNE embeddings and ROC curves).
