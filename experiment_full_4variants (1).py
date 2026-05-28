
import os, sys, pickle, urllib.request
import pickle as pkl
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
from torch.fft import fft, ifft
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

@dataclass
class Config:
    dataset: str = "cora"
    data_dir: str = "data"
    output_dir: str = "visual_outputs"
    seed: int = 0
    hd_dim: int = 10000
    num_updates: int = 2
    include_input: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    chunk_size: int = 256
    supervised_score_batch_size: int = 128
    use_train_val_for_embedding: bool = True
    phi_list: tuple = ("phi1", "phi2", "phi3", "phi4")
    edge_operator: str = "hadamard"

def parse_index_file(filename):
    return [int(line.strip()) for line in open(filename)]

def download_planetoid_dataset(dataset="cora", data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"
    for name in ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]:
        filename = f"ind.{dataset}.{name}"
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename} ...")
            urllib.request.urlretrieve(f"{base_url}/{filename}", path)
        else:
            print(f"{filename} already exists.")

def load_data(dataset="cora", data_dir="data"):
    objects = []
    for name in ["x", "tx", "allx", "graph"]:
        with open(os.path.join(data_dir, f"ind.{dataset}.{name}"), "rb") as f:
            objects.append(pkl.load(f, encoding="latin1") if sys.version_info > (3, 0) else pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_dir, f"ind.{dataset}.test.index"))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset == "citeseer":
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_reorder), :] = tx
        tx = tx_extended
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    return adj, features

def sparse_to_tuple(sparse_mx):
    sparse_mx = sparse_mx.tocoo() if not sp.isspmatrix_coo(sparse_mx) else sparse_mx
    return np.vstack((sparse_mx.row, sparse_mx.col)).transpose(), sparse_mx.data, sparse_mx.shape

def mask_test_edges(adj, seed=0):
    np.random.seed(seed)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj_triu = sp.triu(adj)
    edges = sparse_to_tuple(adj_triu)[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.0))
    num_val = int(np.floor(edges.shape[0] / 20.0))
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:num_val + num_test]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        i, j = np.random.randint(0, adj.shape[0]), np.random.randint(0, adj.shape[0])
        if i == j or ismember([i, j], edges_all): continue
        if test_edges_false and (ismember([j, i], np.array(test_edges_false)) or ismember([i, j], np.array(test_edges_false))): continue
        test_edges_false.append([i, j])
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        i, j = np.random.randint(0, adj.shape[0]), np.random.randint(0, adj.shape[0])
        if i == j: continue
        if ismember([i, j], train_edges) or ismember([j, i], train_edges) or ismember([i, j], val_edges) or ismember([j, i], val_edges): continue
        if val_edges_false and (ismember([j, i], np.array(val_edges_false)) or ismember([i, j], np.array(val_edges_false))): continue
        val_edges_false.append([i, j])
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, train_edges, val_edges, np.array(val_edges_false), test_edges, np.array(test_edges_false)

def row_normalize_dense_features(features):
    row_sum = torch.sum(features, dim=1, keepdim=True)
    row_sum = torch.where(row_sum == 0, torch.tensor(1e-8, dtype=row_sum.dtype, device=row_sum.device), row_sum)
    return features / row_sum

def rp_sign_projection(features, out_dim, seed=0, device="cpu"):
    torch.manual_seed(seed)
    random_A = torch.randn(features.shape[1], out_dim, device=device)
    projected = torch.sign(features.to(device) @ random_A)
    projected[projected == 0] = 1
    return projected

def scipy_sparse_to_torch_sparse(mx, device="cpu"):
    mx = mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values = torch.from_numpy(mx.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(mx.shape), device=device).coalesce()

def circular_bind(x, y): return torch.real(ifft(fft(x, dim=1) * fft(y, dim=1), dim=1))
def rho(X, shift): return torch.roll(X, shifts=shift, dims=1)
def sign_norm(X):
    X = torch.sign(X); X[X == 0] = 1; return X

def gvfa_update(H, Adj_torch_sparse, phi, level):
    F_i = torch.sparse.mm(Adj_torch_sparse, H)
    if phi == "phi1": H_next = H + rho(F_i, level)
    elif phi == "phi2": H_next = H + circular_bind(H, rho(F_i, level))
    elif phi == "phi3": H_next = rho(H + F_i, level)
    elif phi == "phi4": H_next = rho(H + circular_bind(H, F_i), level)
    else: raise ValueError("phi must be phi1, phi2, phi3, phi4")
    return sign_norm(H_next)

def generate_gvfa_embeddings(features, adj_for_embedding, phi, hd_dim, num_updates, include_input, seed, device):
    features = row_normalize_dense_features(features.to(device))
    H0 = rp_sign_projection(features, hd_dim, seed, device)
    Adj = scipy_sparse_to_torch_sparse(adj_for_embedding, device)
    hidden_states, H = [H0], H0
    for level in range(1, num_updates + 1):
        H = gvfa_update(H, Adj, phi, level); hidden_states.append(H)
    Z = torch.cat(hidden_states, dim=1) if include_input else hidden_states[-1]
    return sign_norm(Z), hidden_states

def edge_operator_features(Z, edges, operator="hadamard"):
    out = []
    for u, v in edges:
        z_u, z_v = Z[int(u)], Z[int(v)]
        if operator == "hadamard": r = z_u * z_v
        elif operator == "average": r = (z_u + z_v) / 2.0
        elif operator == "weighted_l1": r = torch.abs(z_u - z_v)
        elif operator == "weighted_l2": r = (z_u - z_v) ** 2
        else: raise ValueError("bad operator")
        out.append(r)
    return torch.stack(out, dim=0)

def sample_negative_edges_from_adj(adj_reference, num_samples, seed=0):
    rng, n, adj_dense, sampled = np.random.default_rng(seed), adj_reference.shape[0], adj_reference.toarray(), set()
    while len(sampled) < num_samples:
        i, j = rng.integers(0, n), rng.integers(0, n)
        if i == j or adj_dense[i, j] != 0: continue
        e = tuple(sorted((int(i), int(j))))
        if e not in sampled: sampled.add(e)
    return np.array(list(sampled))

def evaluate_edge_scores(pos_edges, neg_edges, score_matrix):
    pos = [score_matrix[e[0], e[1]] for e in pos_edges]
    neg = [score_matrix[e[0], e[1]] for e in neg_edges]
    return roc_auc_score(np.hstack([np.ones(len(pos)), np.zeros(len(neg))]), np.hstack([pos, neg])), average_precision_score(np.hstack([np.ones(len(pos)), np.zeros(len(neg))]), np.hstack([pos, neg]))

def evaluate_edge_probabilities(pos_probs, neg_probs):
    return roc_auc_score(np.hstack([np.ones(len(pos_probs)), np.zeros(len(neg_probs))]), np.hstack([pos_probs, neg_probs])), average_precision_score(np.hstack([np.ones(len(pos_probs)), np.zeros(len(neg_probs))]), np.hstack([pos_probs, neg_probs]))

def hit_rate_at_k(edges_pos, edges_neg, score_matrix, k=100):
    scores, labels = [], []
    for e in edges_pos: scores.append(score_matrix[e[0], e[1]]); labels.append(1)
    for e in edges_neg: scores.append(score_matrix[e[0], e[1]]); labels.append(0)
    scores, labels = np.array(scores), np.array(labels)
    top = np.argsort(scores)[::-1][:min(k, len(scores))]
    return np.sum(labels[top] == 1) / min(k, len(edges_pos))

def compute_raw_similarity_scores(Z, chunk_size=256):
    N, D = Z.shape
    score_matrix = torch.empty((N, N), device=Z.device)
    Zf = Z.float()
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        score_matrix[start:end] = ((Zf[start:end] @ Zf.T) / D + 1.0) / 2.0
    return score_matrix

def build_edge_hypervector(Z, edges, device="cpu"):
    edge_hv = torch.zeros(Z.shape[1], device=device)
    for u, v in edges: edge_hv += torch.sign(Z[int(u)] * Z[int(v)])
    edge_hv = torch.sign(edge_hv); edge_hv[edge_hv == 0] = 1
    return edge_hv

def compute_gvfa_hd_link_prediction_scores(Z, pos_edge_hv, neg_edge_hv, chunk_size=256):
    N, D = Z.shape
    score_matrix = torch.empty((N, N), device=Z.device)
    Z_plus, Z_minus = Z * pos_edge_hv, Z * neg_edge_hv
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        D_plus = torch.cdist(Z_plus[start:end].float(), Z.float()) / D
        D_minus = torch.cdist(Z_minus[start:end].float(), Z.float()) / D
        raw = torch.where(D_plus < D_minus, (1.0 - D_plus) + D_minus, D_plus - (1.0 - D_minus))
        score_matrix[start:end] = torch.sigmoid(raw)
    return score_matrix

def train_supervised_edge_classifier(Z, train_pos_edges, train_neg_edges, test_pos_edges, test_neg_edges, operator="hadamard", model_type="logistic_regression"):
    X_train = torch.cat([edge_operator_features(Z, train_pos_edges, operator), edge_operator_features(Z, train_neg_edges, operator)], dim=0).cpu().numpy()
    y_train = np.hstack([np.ones(len(train_pos_edges)), np.zeros(len(train_neg_edges))])
    X_test = torch.cat([edge_operator_features(Z, test_pos_edges, operator), edge_operator_features(Z, test_neg_edges, operator)], dim=0).cpu().numpy()
    if model_type == "logistic_regression":
        clf = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", LogisticRegression(max_iter=100, solver="saga"))])
    else:
        clf = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=0))])
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    pos_probs, neg_probs = probs[:len(test_pos_edges)], probs[len(test_pos_edges):]
    auc, ap = evaluate_edge_probabilities(pos_probs, neg_probs)
    labels = np.hstack([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
    top = np.argsort(probs)[::-1][:100]
    hr100 = np.sum(labels[top] == 1) / min(100, len(pos_probs))
    return auc, ap, hr100, clf

def edge_operator_features_numpy(Z_np, u_indices, v_indices):
    return Z_np[u_indices] * Z_np[v_indices]

def compute_supervised_score_matrix(classifier, Z, batch_size=128):
    Z_np = Z.detach().cpu().float().numpy()
    n = Z_np.shape[0]
    scores = np.zeros((n, n), dtype=np.float32)
    all_v = np.arange(n, dtype=np.int64)
    for u in range(n):
        if u % 100 == 0: print(f"    supervised score row {u}/{n}")
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            v_batch = all_v[start:end]
            u_batch = np.full(len(v_batch), u, dtype=np.int64)
            X = edge_operator_features_numpy(Z_np, u_batch, v_batch)
            scores[u, start:end] = classifier.predict_proba(X)[:, 1].astype(np.float32)
    return scores

def make_prediction_table(pos_edges, neg_edges, score_matrix):
    rows = []
    for e in pos_edges:
        u, v = int(e[0]), int(e[1]); rows.append({"u": u, "v": v, "label": 1, "edge_type": "hidden_positive", "score": float(score_matrix[u, v])})
    for e in neg_edges:
        u, v = int(e[0]), int(e[1]); rows.append({"u": u, "v": v, "label": 0, "edge_type": "sampled_negative", "score": float(score_matrix[u, v])})
    rows = sorted(rows, key=lambda x: x["score"], reverse=True)
    for rank, row in enumerate(rows, start=1): row["rank"] = rank
    return rows

def save_gvfa_visual_package_4variants(output_dir, phi, adj_train, adj_for_embedding, train_edges, val_edges, test_edges, test_edges_false, hidden_states, Z, raw_score_np, hd_score_np, lr_score_np, mlp_score_np, auc_v1, ap_v1, hr_v1, auc_v2, ap_v2, hr_v2, auc_v3, ap_v3, hr_v3, auc_v4, ap_v4, hr_v4):
    os.makedirs(output_dir, exist_ok=True)
    train_graph_edges = np.array(adj_train.nonzero()).T
    train_graph_edges = train_graph_edges[train_graph_edges[:, 0] < train_graph_edges[:, 1]]
    model_graph_edges = np.array(adj_for_embedding.nonzero()).T
    model_graph_edges = model_graph_edges[model_graph_edges[:, 0] < model_graph_edges[:, 1]]
    hidden_states_preview = [H[:500, :128].detach().cpu().numpy().astype(int).tolist() for H in hidden_states]
    Z_preview = Z[:500, :128].detach().cpu().numpy().astype(int).tolist()
    raw_score_np, hd_score_np, lr_score_np, mlp_score_np = map(lambda x: np.asarray(x, dtype=np.float32), [raw_score_np, hd_score_np, lr_score_np, mlp_score_np])
    package = {
        "phi": phi, "num_nodes": int(adj_train.shape[0]), "num_train_edges": int(len(train_edges)), "num_val_edges": int(len(val_edges)), "num_test_edges": int(len(test_edges)),
        "selected_nodes": list(range(500)), "graph_edges": model_graph_edges.tolist(), "model_graph_edges": model_graph_edges.tolist(), "train_graph_edges": train_graph_edges.tolist(),
        "train_edges": train_edges.tolist(), "val_edges": val_edges.tolist(), "test_edges": test_edges.tolist(), "test_edges_false": test_edges_false.tolist(),
        "hidden_states_preview": hidden_states_preview, "Z_preview": Z_preview,
        "raw_score_preview": raw_score_np, "hd_score_preview": hd_score_np, "lr_score_preview": lr_score_np, "mlp_score_preview": mlp_score_np,
        "raw_top_predictions": make_prediction_table(test_edges, test_edges_false, raw_score_np), "hd_top_predictions": make_prediction_table(test_edges, test_edges_false, hd_score_np),
        "lr_top_predictions": make_prediction_table(test_edges, test_edges_false, lr_score_np), "mlp_top_predictions": make_prediction_table(test_edges, test_edges_false, mlp_score_np),
        "metrics": {
            "variant_1_raw_similarity": {"auc": float(auc_v1), "ap": float(ap_v1), "hr100": float(hr_v1)},
            "variant_2_gvfa_hd": {"auc": float(auc_v2), "ap": float(ap_v2), "hr100": float(hr_v2)},
            "variant_3_hadamard_lr": {"auc": float(auc_v3), "ap": float(ap_v3), "hr100": float(hr_v3)},
            "variant_4_hadamard_mlp": {"auc": float(auc_v4), "ap": float(ap_v4), "hr100": float(hr_v4)},
        },
    }
    out = os.path.join(output_dir, f"gvfa_visual_{phi}.pkl")
    with open(out, "wb") as f: pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved:", out)

def main():
    cfg = Config(); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    print(f"Using device: {cfg.device}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    download_planetoid_dataset(cfg.dataset, cfg.data_dir)
    adj, features = load_data(cfg.dataset, cfg.data_dir)
    adj_orig = adj.copy(); adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape); adj_orig.eliminate_zeros()
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_orig, seed=cfg.seed)
    num_nodes = adj_orig.shape[0]
    train_val_edges = np.vstack([train_edges, val_edges])
    data = np.ones(train_val_edges.shape[0])
    adj_train_val = sp.csr_matrix((data, (train_val_edges[:, 0], train_val_edges[:, 1])), shape=(num_nodes, num_nodes))
    adj_train_val = adj_train_val + adj_train_val.T
    adj_train_val = adj_train_val - sp.dia_matrix((adj_train_val.diagonal()[np.newaxis, :], [0]), shape=adj_train_val.shape); adj_train_val.eliminate_zeros()
    adj_for_embedding, positive_edges_for_hv = (adj_train_val, train_val_edges) if cfg.use_train_val_for_embedding else (adj_train, train_edges)
    features = torch.tensor(features.toarray()).float()
    all_results = []
    for phi in cfg.phi_list:
        print("\n" + "=" * 80); print(f"GVFA configuration: {phi.upper()}"); print("=" * 80)
        Z, hidden_states = generate_gvfa_embeddings(features, adj_for_embedding, phi, cfg.hd_dim, cfg.num_updates, cfg.include_input, cfg.seed, cfg.device)
        print("Node embedding shape:", tuple(Z.shape))
        raw_score_matrix = compute_raw_similarity_scores(Z, cfg.chunk_size); raw_score_np = raw_score_matrix.detach().cpu().numpy().astype(np.float32)
        auc_v1, ap_v1 = evaluate_edge_scores(test_edges, test_edges_false, raw_score_np); hr_v1 = hit_rate_at_k(test_edges, test_edges_false, raw_score_np)
        print(f"Variant 1 - Raw similarity | AUC: {auc_v1:.4f}, AP: {ap_v1:.4f}, HR@100: {hr_v1:.4f}")
        pos_edge_hv = build_edge_hypervector(Z, positive_edges_for_hv, cfg.device)
        extra_negative_edges = sample_negative_edges_from_adj(adj_train_val, len(val_edges_false), cfg.seed)
        neg_edge_hv = build_edge_hypervector(Z, np.vstack([val_edges_false, extra_negative_edges]), cfg.device)
        hd_score_matrix = compute_gvfa_hd_link_prediction_scores(Z, pos_edge_hv, neg_edge_hv, cfg.chunk_size); hd_score_np = hd_score_matrix.detach().cpu().numpy().astype(np.float32)
        auc_v2, ap_v2 = evaluate_edge_scores(test_edges, test_edges_false, hd_score_np); hr_v2 = hit_rate_at_k(test_edges, test_edges_false, hd_score_np)
        print(f"Variant 2 - GVFA HD LP | AUC: {auc_v2:.4f}, AP: {ap_v2:.4f}, HR@100: {hr_v2:.4f}")
        train_neg_edges = sample_negative_edges_from_adj(adj_train_val, len(train_edges), cfg.seed + 11)
        auc_v3, ap_v3, hr_v3, lr_model = train_supervised_edge_classifier(Z, train_edges, train_neg_edges, test_edges, test_edges_false, cfg.edge_operator, "logistic_regression")
        print(f"Variant 3 - Hadamard + LR | AUC: {auc_v3:.4f}, AP: {ap_v3:.4f}, HR@100: {hr_v3:.4f}")
        auc_v4, ap_v4, hr_v4, mlp_model = train_supervised_edge_classifier(Z, train_edges, train_neg_edges, test_edges, test_edges_false, cfg.edge_operator, "mlp")
        print(f"Variant 4 - Hadamard + MLP | AUC: {auc_v4:.4f}, AP: {ap_v4:.4f}, HR@100: {hr_v4:.4f}")
        print("Computing full LR score matrix for dashboard..."); lr_score_np = compute_supervised_score_matrix(lr_model, Z, cfg.supervised_score_batch_size)
        print("Computing full MLP score matrix for dashboard..."); mlp_score_np = compute_supervised_score_matrix(mlp_model, Z, cfg.supervised_score_batch_size)
        print("Saving 4-variant visualization package...")
        save_gvfa_visual_package_4variants(cfg.output_dir, phi, adj_train, adj_for_embedding, train_edges, val_edges, test_edges, test_edges_false, hidden_states, Z, raw_score_np, hd_score_np, lr_score_np, mlp_score_np, auc_v1, ap_v1, hr_v1, auc_v2, ap_v2, hr_v2, auc_v3, ap_v3, hr_v3, auc_v4, ap_v4, hr_v4)
        all_results.append({"phi": phi, "embedding_shape": tuple(Z.shape), "variant_1_raw_similarity_auc": auc_v1, "variant_2_gvfa_hd_auc": auc_v2, "variant_3_hadamard_lr_auc": auc_v3, "variant_4_hadamard_mlp_auc": auc_v4})
        del Z, hidden_states, raw_score_matrix, hd_score_matrix, raw_score_np, hd_score_np, lr_score_np, mlp_score_np, lr_model, mlp_model
        if cfg.device == "cuda": torch.cuda.empty_cache()
    print("\nFinal Results")
    for row in all_results: print(row)

if __name__ == "__main__": main()
