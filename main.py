# Imports
# --------------------------
import dgl
print("dgl version: " + dgl.__version__)
import torch
print("PyTorch version: " + torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
print("NetworkX version: " + nx.__version__)
import numpy as np
print("Numpy version: " + np.__version__)
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.manifold import TSNE
from dgl.data import CoraGraphDataset
from node2vec import Node2Vec
from dgl.nn import GraphConv
import random
import matplotlib.pyplot as plt


# Ρύθμιση του seed
# --------------------------
def set_seed(seed=42):
    # Ρύθμιση seed για επαναληψιμότητα.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


class Visualizer:
    # Κλάση υπεύθυνη για όλες τις οπτικοποιήσεις.

    @staticmethod
    def plot_graph_structure(graph_nx, labels, title="Graph Structure"):
        # Σχεδιάζει ένα υπο-γράφημα (γειτονιά ενός hub node).
        print(f"   [Plotting] {title}...")
        degrees = dict(graph_nx.degree())
        center_node = max(degrees, key=degrees.get)
        subgraph = nx.ego_graph(graph_nx, center_node, radius=2)

        if subgraph.number_of_nodes() > 100:
            subgraph = nx.ego_graph(graph_nx, center_node, radius=1)

        node_colors = [labels[n].item() for n in subgraph.nodes()]

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(subgraph, seed=42, k=0.15)
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, cmap=plt.cm.Set2, node_size=120, alpha=0.9)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.3, edge_color='gray')
        plt.title(f"{title}\n(Neighborhood of Node {center_node})")
        plt.axis('off')
        plt.show()

    @staticmethod
    def plot_training_loss(losses, title="Training Loss"):
        # Σχεδιάζει την καμπύλη του Loss.
        plt.figure(figsize=(8, 5))
        plt.plot(losses, label='Training Loss', color='purple')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    @staticmethod
    def plot_tsne(embeddings, labels, title="t-SNE Embeddings"):
        # Οπτικοποιεί τα embeddings σε 2D χώρο.
        print(f"   [Plotting] Generating t-SNE for {title}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
        emb_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', s=15, alpha=0.7)
        plt.title(title)
        plt.colorbar(scatter, label='Class Label')
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def plot_roc_comparison(results_dict):
        # Συγκρίνει τις καμπύλες ROC όλων των μεθόδων.
        plt.figure(figsize=(10, 8))
        for name, data in results_dict.items():
            y_true, y_scores = data['y_true'], data['y_scores']
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.5)
        plt.show()


# 1. Προετοιμασία και Επεξεργασία Δεδομένων
# --------------------------

def prepare_data():
    print("\n 1. Προετοιμασία Δεδομένων")
    dataset = CoraGraphDataset()
    g = dataset[0]

    # Καθαρισμός γραφήματος [cite: 97]
    g = dgl.remove_self_loop(g)
    g = dgl.to_simple(g)

    # Έλεγχος Συνεκτικότητας (Connectivity) & LCC [cite: 96]
    nx_temp = dgl.to_networkx(g).to_undirected()
    if not nx.is_connected(nx_temp):
        print("\nWarning: Μη συνεκτικό γράφημα. Κρατάμε το Largest Connected Component (LCC).")
        largest_cc = max(nx.connected_components(nx_temp), key=len)
        g = dgl.node_subgraph(g, list(largest_cc))

    nx_g = nx.Graph(dgl.to_networkx(g, node_attrs=['feat', 'label']))
    node_labels = g.ndata['label']

    # Διαχωρισμός Train/Test (10%) με στρατηγική MST για διατήρηση συνεκτικότητας [cite: 96]
    all_edges = list(nx_g.edges())
    num_test = int(len(all_edges) * 0.1)

    # Βρίσκουμε το MST που ΠΡΕΠΕΙ να μείνει στο Train set
    mst_edges = set(tuple(sorted((u, v))) for u, v in nx.minimum_spanning_edges(nx_g, data=False))

    # Υποψήφιες ακμές για το Test set είναι αυτές που δεν ανήκουν στο MST
    candidates = [e for e in all_edges if tuple(sorted(e)) not in mst_edges]
    random.shuffle(candidates)

    test_pos_edges = candidates[:num_test]
    # Το train set αποτελείται από το MST + τις υπόλοιπες ακμές που δεν επιλέχθηκαν για test
    train_edges = list(mst_edges) + candidates[num_test:]

    # Κατασκευή Training Graph
    train_g_nx = nx.Graph()
    train_g_nx.add_nodes_from(nx_g.nodes(data=True))
    train_g_nx.add_edges_from(train_edges)

    # Μετατροπή σε DGL για το GNN
    train_g_dgl = dgl.from_networkx(train_g_nx)
    train_g_dgl.ndata['feat'] = g.ndata['feat']

    # Negative Sampling για το Test Set (1:1 ratio) [cite: 99-100]
    test_neg_edges = get_negative_samples(nx_g, len(test_pos_edges))

    print(f"Nodes (LCC): {train_g_nx.number_of_nodes()}")
    print(f"Total Edges: {len(all_edges)}")
    print(f"Train Edges: {len(train_edges)}")
    print(f"Test Edges: {len(test_pos_edges)}")

    return g, train_g_dgl, train_g_nx, test_pos_edges, test_neg_edges, node_labels


def get_negative_samples(graph_nx, num_samples):
    # Επιλέγει τυχαία ζεύγη κόμβων που δεν συνδέονται μεταξύ τους.
    neg_edges = set()
    nodes = list(graph_nx.nodes())
    while len(neg_edges) < num_samples:
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u != v and not graph_nx.has_edge(u, v):
            neg_edges.add(tuple(sorted((u, v))))
    return list(neg_edges)


# 2. Ευριστικές Μέθοδοι
# --------------------------

def evaluate_heuristics(train_nx, test_pos, test_neg, results_collector):
    print("\n 2. Μέρος Α: Ευριστικές Μέθοδοι")

    # Ορισμός συναρτήσεων βάσει των τύπων του PDF
    methods = {
        # Jaccard = |N(u) n N(v)| / |N(u) u N(v)|
        "Jaccard": lambda u, v: list(nx.jaccard_coefficient(train_nx, [(u, v)]))[0][2],

        # Adamic-Adar = sum(1 / log(|N(w)|)) for w in common_neighbors [cite: 112]
        "Adamic-Adar": lambda u, v: list(nx.adamic_adar_index(train_nx, [(u, v)]))[0][2],

        # Common Neighbors = |N(u) n N(v)| [cite: 110]
        "Common Neighbors": lambda u, v: len(list(nx.common_neighbors(train_nx, u, v)))
    }

    test_edges = test_pos + test_neg
    y_true = [1] * len(test_pos) + [0] * len(test_neg)

    for name, func in methods.items():
        y_scores = []
        for u, v in test_edges:
            try:
                score = func(u, v)
            except:
                score = 0
            y_scores.append(score)

        auc_score = roc_auc_score(y_true, y_scores)
        print(f"   {name} AUC: {auc_score:.4f}")

        # Αποθήκευση για ROC plot
        results_collector[name] = {'y_true': y_true, 'y_scores': y_scores}


# 3. Node2Vec
# --------------------------

class LinkPredMLP(nn.Module):
    # Απλός ταξινομητής MLP (1-2 επίπεδα)[cite: 124].

    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


def part_b_node2vec(full_nx, train_nx, test_pos, test_neg, labels, results_collector):
    print("\n 3. Μέρος Β: Node2Vec & Shallow Embeddings")

    # 1. Οπτικοποίηση Δομής
    Visualizer.plot_graph_structure(full_nx, labels, title="Graph Structure (Sample)")

    # 2. Εκπαίδευση Node2Vec στο ΠΛΗΡΕΣ γράφημα (Unsupervised) [cite: 119]
    print("   Training Node2Vec on full graph...")
    n2v = Node2Vec(full_nx, dimensions=64, walk_length=10, num_walks=20, workers=1, quiet=True)
    model = n2v.fit(window=10, min_count=1)

    # Εξαγωγή Embeddings
    node_ids = sorted([n for n in full_nx.nodes()])
    embeddings_matrix = np.array([model.wv[str(n)] for n in node_ids])
    embeddings_dict = {n: embeddings_matrix[i] for i, n in enumerate(node_ids)}

    # 3. Οπτικοποίηση Embeddings με t-SNE
    Visualizer.plot_tsne(embeddings_matrix, labels.numpy(), title="Node2Vec Embeddings")

    # 4. Προετοιμασία MLP
    # Χρήση Hadamard product για features ακμών: h_uv = z_u * z_v
    def get_edge_feats(edges):
        feats = []
        for u, v in edges:
            emb_u = embeddings_dict[u] if u in embeddings_dict else np.zeros(64)
            emb_v = embeddings_dict[v] if v in embeddings_dict else np.zeros(64)
            feats.append(emb_u * emb_v)
        return torch.FloatTensor(np.array(feats))

    # Training Data για MLP (από το train_nx μόνο) [cite: 125]
    tr_pos = list(train_nx.edges())
    tr_neg = get_negative_samples(train_nx, len(tr_pos))
    X_train = torch.cat([get_edge_feats(tr_pos), get_edge_feats(tr_neg)])
    y_train = torch.cat([torch.ones(len(tr_pos)), torch.zeros(len(tr_neg))]).unsqueeze(1)

    mlp = LinkPredMLP(64)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    print(" Training MLP Classifier...")
    losses = []
    epochs = 50
    for e in range(epochs):
        optimizer.zero_grad()
        preds = mlp(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    Visualizer.plot_training_loss(losses, title="Node2Vec+MLP Training Loss")

    # 5. Αξιολόγηση στο Test Set
    X_test = torch.cat([get_edge_feats(test_pos), get_edge_feats(test_neg)])
    y_test_true = torch.cat([torch.ones(len(test_pos)), torch.zeros(len(test_neg))]).numpy()

    with torch.no_grad():
        y_test_scores = mlp(X_test).squeeze().numpy()
        auc_score = roc_auc_score(y_test_true, y_test_scores)
        print(f"   Node2Vec + MLP AUC: {auc_score:.4f}")

        results_collector["Node2Vec+MLP"] = {'y_true': y_test_true, 'y_scores': y_test_scores}


# 4. End-To-End GNN
# --------------------------

class GCN(nn.Module):
    # GCN Encoder που χαρτογραφεί u -> z_u [cite: 131-132].

    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)

    def forward(self, g, in_feat):
        h = F.relu(self.conv1(g, in_feat))
        h = self.conv2(g, h)
        return h


class DotPredictor(nn.Module):
    # Predictor με Dot Product: y_uv = sigmoid(z_u^T * z_v)[cite: 136].

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]


def part_c_gnn(train_g, train_nx, test_pos, test_neg, feats, labels, results_collector):
    print("\n 4. Μέρος Γ: End-to-End GN")

    model = GCN(feats.shape[1], 16)
    pred = DotPredictor()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(pred.parameters()), lr=0.01)

    losses = []
    print(" Training GNN End-to-End...")

    # Training Loop [cite: 138-141]
    for e in range(101):
        # Online Negative Sampling σε κάθε epoch
        neg_edges = get_negative_samples(train_nx, train_g.num_edges())
        u_neg, v_neg = zip(*neg_edges)
        neg_g = dgl.graph((torch.tensor(u_neg), torch.tensor(v_neg)), num_nodes=train_g.num_nodes())

        # Forward Pass
        g_loop = dgl.add_self_loop(train_g)
        h = model(g_loop, feats)  # Παράγει embeddings z_u

        pos_score = pred(train_g, h)
        neg_score = pred(neg_g, h)

        # Binary Cross Entropy Loss [cite: 139]
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_score, neg_score]),
            torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if e % 20 == 0:
            print(f"     Epoch {e}, Loss: {loss.item():.4f}")

    Visualizer.plot_training_loss(losses, title="GNN End-to-End Training Loss")

    # Οπτικοποίηση των GNN Embeddings μετά την εκπαίδευση
    with torch.no_grad():
        final_h = model(dgl.add_self_loop(train_g), feats).numpy()
        Visualizer.plot_tsne(final_h, labels.numpy(), title="GNN Learned Embeddings")

    # Evaluation στο Test Set
    u_test, v_test = zip(*test_pos)
    test_pos_g = dgl.graph((torch.tensor(u_test), torch.tensor(v_test)), num_nodes=train_g.num_nodes())
    u_neg, v_neg = zip(*test_neg)
    test_neg_g = dgl.graph((torch.tensor(u_neg), torch.tensor(v_neg)), num_nodes=train_g.num_nodes())

    with torch.no_grad():
        h = model(dgl.add_self_loop(train_g), feats)
        pos_scores = pred(test_pos_g, h)
        neg_scores = pred(test_neg_g, h)

        y_test_scores = torch.cat([pos_scores, neg_scores]).sigmoid().numpy()
        y_test_true = torch.cat([torch.ones(len(pos_scores)), torch.zeros(len(neg_scores))]).numpy()

        auc_score = roc_auc_score(y_test_true, y_test_scores)
        print(f"   GNN End-to-End AUC: {auc_score:.4f}")

        results_collector["GNN (GCN)"] = {'y_true': y_test_true, 'y_scores': y_test_scores}


# Main
# --------------------------
def main():
    # Λεξικό για αποθήκευση αποτελεσμάτων για συγκριτικό plot
    results = {}

    # 1. Προετοιμασία
    full_dgl_g, train_g_dgl, train_g_nx, test_pos, test_neg, node_labels = prepare_data()

    # 2. Μέρος Α: Heuristics
    evaluate_heuristics(train_g_nx, test_pos, test_neg, results)

    # 3. Μέρος Β: Node2Vec
    part_b_node2vec(nx.Graph(dgl.to_networkx(full_dgl_g)), train_g_nx, test_pos, test_neg, node_labels, results)

    # 4. Μέρος Γ: GNN
    part_c_gnn(train_g_dgl, train_g_nx, test_pos, test_neg, full_dgl_g.ndata['feat'], node_labels, results)

    # 5. Τελική Σύγκριση
    print("\n Τελική Σύγκριση & Οπτικοποίηση ROC")
    Visualizer.plot_roc_comparison(results)


if __name__ == "__main__":
    main()

# END 407