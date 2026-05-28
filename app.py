
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="GVFA Link Prediction Explorer",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONSTANTS / VARIANT REGISTRY
# ============================================================
DEFAULT_THRESHOLD = 0.5

PLOTLY_GRAPH_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "gvfa_cora_pair_query_3d_graph",
        "height": 1000,
        "width": 1800,
        "scale": 2,
    },
}

COLORS = {
    "train": "rgba(170,170,170,0.13)",
    "train_focus": "rgba(120,120,120,0.42)",
    "TP": "rgba(46,204,113,0.95)",
    "FP": "rgba(255,140,0,0.95)",
    "TN": "rgba(52,152,219,0.32)",
    "FN": "rgba(231,76,60,0.95)",
    "query_link": "black",
    "query_no_link": "crimson",
    "node": "rgba(185,185,185,0.58)",
    "node_focus": "rgba(255,215,0,0.95)",
    "node_u": "cyan",
    "node_v": "orange",
    "query_marker": "yellow",
}

CONFUSION_LABELS = {
    "TP": "TP - True Positive",
    "FP": "FP - False Positive",
    "TN": "TN - True Negative",
    "FN": "FN - False Negative",
}

VARIANTS = {
    "raw": {
        "title": "Variant 1 — RAW Similarity",
        "short": "RAW Similarity",
        "subtitle": "Raw GVFA similarity score pipeline",
        "badge": "variant-1 = raw",
        "score_keys": ["raw_score_preview"],
        "metrics_key": "variant_1_raw_similarity",
    },
    "hd": {
        "title": "Variant 2 — GVFA HD",
        "short": "GVFA HD",
        "subtitle": "Positive/negative edge hypervector pipeline",
        "badge": "variant-2 = gvfa-hd",
        "score_keys": ["hd_score_preview"],
        "metrics_key": "variant_2_gvfa_hd",
    },
    "lr": {
        "title": "Variant 3 — Hadamard LR",
        "short": "Hadamard LR",
        "subtitle": "Hadamard edge features + Logistic Regression",
        "badge": "variant-3 = hadamard-lr",
        "score_keys": ["lr_score_preview", "hadamard_lr_score_preview", "variant_3_score_preview"],
        "metrics_key": "variant_3_hadamard_lr",
    },
    "mlp": {
        "title": "Variant 4 — Hadamard MLP",
        "short": "Hadamard MLP",
        "subtitle": "Hadamard edge features + MLP classifier",
        "badge": "variant-4 = hadamard-mlp",
        "score_keys": ["mlp_score_preview", "hadamard_mlp_score_preview", "variant_4_score_preview"],
        "metrics_key": "variant_4_hadamard_mlp",
    },
}

STATUS_BADGE_COLORS = {
    "TP": "#2ecc71",
    "FP": "#ff8c00",
    "TN": "#3498db",
    "FN": "#e74c3c",
}


# ============================================================
# STYLING
# ============================================================
def inject_css():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
        .block-container {
            padding-top: 1.1rem;
            padding-left: 1.6rem;
            padding-right: 1.6rem;
            max-width: 100%;
        }
        .comparison-card {
            border: 1px solid rgba(49, 97, 255, 0.12);
            border-radius: 18px;
            padding: 18px 20px;
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,255,0.96));
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            min-height: 165px;
            margin-bottom: 14px;
        }
        .flow-title {font-size: 19px; font-weight: 800; color: #2563eb; margin-bottom: 6px;}
        .flow-subtitle {font-size: 13px; color: #667085; margin-bottom: 12px;}
        .pill-row {display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 12px 0;}
        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 12px;
            font-weight: 700;
            color: #344054;
            background: #f1f3ff;
            border: 1px solid #d8dcff;
        }
        .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 6px 12px;
            color: white;
            font-size: 13px;
            font-weight: 800;
        }
        .kv-grid {
            display: grid;
            grid-template-columns: 105px 1fr;
            row-gap: 5px;
            column-gap: 8px;
            font-size: 13px;
        }
        .kv-key {color: #667085; font-weight: 700;}
        .kv-val {color: #111827; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;}
        .graph-card-title {font-weight: 800; color: #475467; margin: 8px 0 4px 0;}
        .small-caption {font-size: 13px; color: #667085; margin-top: -8px; margin-bottom: 8px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_resource(show_spinner=False)
def load_package(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_phi_files():
    folder = "visual_outputs"
    files = {}
    if not os.path.exists(folder):
        return files
    for fname in os.listdir(folder):
        if fname.startswith("gvfa_visual_") and fname.endswith(".pkl"):
            phi = fname.replace("gvfa_visual_", "").replace(".pkl", "")
            files[phi] = os.path.join(folder, fname)
    return files


# ============================================================
# BASIC HELPERS
# ============================================================
def edge_key(u, v):
    return tuple(sorted((int(u), int(v))))


def make_edge_set(edge_list):
    return {edge_key(u, v) for u, v in edge_list}


def edge_set_to_sorted_list(edge_set):
    return sorted(list(edge_set), key=lambda x: (x[0], x[1]))


def get_available_score_key(data, variant_key):
    for key in VARIANTS[variant_key]["score_keys"]:
        if key in data:
            return key
    return None


def get_score_matrix(data, variant_key):
    score_key = get_available_score_key(data, variant_key)
    if score_key is None:
        expected = ", ".join(VARIANTS[variant_key]["score_keys"])
        raise KeyError(f"Missing score matrix for {VARIANTS[variant_key]['short']}. Expected one of: {expected}")
    return np.asarray(data[score_key])


def get_prediction(data, variant_key, u, v):
    mat = get_score_matrix(data, variant_key)
    u = int(u)
    v = int(v)
    if 0 <= u < mat.shape[0] and 0 <= v < mat.shape[1]:
        return float(mat[u, v])
    return None


def get_confusion_code(pred_label, actual_label):
    if pred_label == 1 and actual_label == 1:
        return "TP"
    if pred_label == 1 and actual_label == 0:
        return "FP"
    if pred_label == 0 and actual_label == 1:
        return "FN"
    return "TN"


def get_pair_testing_status(u, v, pred_label, full_positive_set):
    edge_status = 1 if edge_key(u, v) in full_positive_set else 0
    code = get_confusion_code(pred_label=pred_label, actual_label=edge_status)
    return code, CONFUSION_LABELS[code], edge_status


def variant_is_available(data, variant_key):
    return get_available_score_key(data, variant_key) is not None


def available_variant_options(data):
    return [k for k in VARIANTS.keys() if variant_is_available(data, k)]


# ============================================================
# GRAPH METADATA + 3D LAYOUT
# ============================================================
@st.cache_data(show_spinner=False)
def compute_graph_metadata(full_cora_edges_tuple, num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(int(num_nodes)))
    G.add_edges_from(full_cora_edges_tuple)

    degree_map = {int(node): int(deg) for node, deg in G.degree()}
    neighbor_map = {int(node): sorted([int(n) for n in G.neighbors(node)]) for node in G.nodes()}

    component_id_map = {}
    component_size_map = {}
    for comp_id, component in enumerate(nx.connected_components(G)):
        component = list(component)
        comp_size = len(component)
        for node in component:
            component_id_map[int(node)] = int(comp_id)
            component_size_map[int(node)] = int(comp_size)

    return degree_map, neighbor_map, component_id_map, component_size_map


@st.cache_data(show_spinner=True)
def compute_full_cora_layout_3d(full_cora_edges_tuple, num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(int(num_nodes)))
    G.add_edges_from(full_cora_edges_tuple)

    pos = nx.spring_layout(
        G,
        dim=3,
        seed=42,
        iterations=45,
        k=1 / np.sqrt(max(int(num_nodes), 1)),
    )

    return {int(node): (float(x), float(y), float(z)) for node, (x, y, z) in pos.items()}


# ============================================================
# FOCUS HELPERS
# ============================================================
def get_focus_nodes(u, v, neighbor_map, hops=1):
    u = int(u)
    v = int(v)
    focus_nodes = {u, v}
    frontier = {u, v}

    for _ in range(int(hops)):
        next_frontier = set()
        for node in frontier:
            for nbr in neighbor_map.get(int(node), []):
                nbr = int(nbr)
                if nbr not in focus_nodes:
                    next_frontier.add(nbr)
        focus_nodes.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    return focus_nodes


def filter_edges_touching_focus(edge_items, focus_nodes):
    focus_nodes = {int(n) for n in focus_nodes}
    filtered = []
    for item in edge_items:
        if isinstance(item, dict):
            u = int(item["u"])
            v = int(item["v"])
        else:
            u, v = item
            u = int(u)
            v = int(v)
        if u in focus_nodes or v in focus_nodes:
            filtered.append(item)
    return filtered


def get_focus_ranges_3d(pos, focus_nodes, padding_ratio=0.85, min_span=0.08):
    xs, ys, zs = [], [], []
    for node in focus_nodes:
        if int(node) in pos:
            x, y, z = pos[int(node)]
            xs.append(float(x)); ys.append(float(y)); zs.append(float(z))

    if not xs:
        return None

    def padded_range(values):
        lo, hi = min(values), max(values)
        span = max(hi - lo, min_span)
        pad = span * padding_ratio
        return [lo - pad, hi + pad]

    return padded_range(xs), padded_range(ys), padded_range(zs)


# ============================================================
# TESTING CONFUSION EDGES FOR OVERLAY
# ============================================================
def build_testing_confusion_edges(data, variant_key, threshold, test_edges, test_edges_false):
    mat = get_score_matrix(data, variant_key)
    groups = {"TP": [], "FP": [], "TN": [], "FN": []}

    for u, v in test_edges:
        u = int(u); v = int(v)
        score = float(mat[u, v])
        pred_label = 1 if score >= threshold else 0
        code = get_confusion_code(pred_label, actual_label=1)
        groups[code].append({"u": u, "v": v, "score": score, "actual": 1, "predicted": pred_label})

    for u, v in test_edges_false:
        u = int(u); v = int(v)
        score = float(mat[u, v])
        pred_label = 1 if score >= threshold else 0
        code = get_confusion_code(pred_label, actual_label=0)
        groups[code].append({"u": u, "v": v, "score": score, "actual": 0, "predicted": pred_label})

    return groups


# ============================================================
# 3D PLOT HELPERS
# ============================================================
def add_edge_trace_3d(fig, edge_items, pos, color, width, name, show_hover=True):
    edge_x, edge_y, edge_z, hover_text = [], [], [], []
    for item in edge_items:
        if isinstance(item, dict):
            u = int(item["u"]); v = int(item["v"])
            score = item.get("score")
            actual = item.get("actual")
            predicted = item.get("predicted")
            h = (
                f"<b>{name}</b><br>u: {u}<br>v: {v}<br>"
                f"Score: {score:.4f}<br>Edge label: {actual}<br>Predicted: {predicted}"
                if score is not None else f"<b>{name}</b><br>u: {u}<br>v: {v}"
            )
        else:
            u, v = item
            u = int(u); v = int(v)
            h = f"<b>{name}</b><br>u: {u}<br>v: {v}"

        if u not in pos or v not in pos:
            continue

        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        hover_text.extend([h, h, None])

    if not edge_x:
        return

    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color=color, width=width),
            hovertext=hover_text if show_hover else None,
            hoverinfo="text" if show_hover else "none",
            name=name,
        )
    )


def add_nodes_trace_3d(fig, pos, num_nodes, u, v, degree_map, neighbor_map, component_id_map, component_size_map, focus_nodes):
    node_x, node_y, node_z = [], [], []
    node_color, node_size, node_hover = [], [], []
    selected_u, selected_v = int(u), int(v)
    focus_nodes = {int(n) for n in focus_nodes}

    for node in range(int(num_nodes)):
        if node not in pos:
            continue

        x, y, z = pos[node]
        deg = degree_map.get(node, 0)
        comp_id = component_id_map.get(node, -1)
        comp_size = component_size_map.get(node, 0)
        neighbors = neighbor_map.get(node, [])

        if node == selected_u:
            role, color, size = "Node u", COLORS["node_u"], 8
        elif node == selected_v:
            role, color, size = "Node v", COLORS["node_v"], 8
        elif node in focus_nodes:
            role, color, size = "Neighbourhood node", COLORS["node_focus"], 4
        else:
            role, color, size = "Cora node", COLORS["node"], 2.3

        hover_text = (
            f"<b>Node {node}</b><br>Role: {role}<br>Degree: {deg}<br>"
            f"Component ID: {comp_id}<br>Component size: {comp_size}<br>"
            f"Neighbour count: {len(neighbors)}<br>First neighbours: {neighbors[:15]}"
        )

        node_x.append(x); node_y.append(y); node_z.append(z)
        node_color.append(color); node_size.append(size); node_hover.append(hover_text)

    fig.add_trace(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            marker=dict(size=node_size, color=node_color, opacity=0.86, line=dict(width=0.5, color="white")),
            hovertext=node_hover,
            hoverinfo="text",
            name="Cora nodes",
            showlegend=False,
        )
    )


def add_query_pair_dashed_trace_3d(fig, pos, u, v, score, threshold, testing_status_text):
    u, v = int(u), int(v)
    if u not in pos or v not in pos:
        return

    x0, y0, z0 = pos[u]
    x1, y1, z1 = pos[v]
    pred_label = 1 if score >= threshold else 0
    query_color = COLORS["query_link"] if pred_label == 1 else COLORS["query_no_link"]
    hover = f"<b>Queried pair</b><br>u: {u}<br>v: {v}<br>Score: {score:.4f}<br>Testing Status: {testing_status_text}"

    dash_count, dash_fraction = 15, 0.55
    dash_x, dash_y, dash_z = [], [], []
    for i in range(dash_count):
        t_start = i / dash_count
        t_end = min(t_start + dash_fraction / dash_count, 1.0)
        xs = x0 + (x1 - x0) * t_start
        ys = y0 + (y1 - y0) * t_start
        zs = z0 + (z1 - z0) * t_start
        xe = x0 + (x1 - x0) * t_end
        ye = y0 + (y1 - y0) * t_end
        ze = z0 + (z1 - z0) * t_end
        dash_x.extend([xs, xe, None])
        dash_y.extend([ys, ye, None])
        dash_z.extend([zs, ze, None])

    fig.add_trace(go.Scatter3d(x=dash_x, y=dash_y, z=dash_z, mode="lines", line=dict(color=query_color, width=10), hovertext=[hover] * len(dash_x), hoverinfo="text", name="Queried pair"))

    fig.add_trace(
        go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1], mode="markers+text",
            marker=dict(size=[8, 8], color=[COLORS["node_u"], COLORS["node_v"]], line=dict(width=3, color="black")),
            text=[f"u={u}", f"v={v}"], textposition="top center", textfont=dict(size=12, color="black"),
            hovertext=[f"<b>Node u</b><br>Node: {u}", f"<b>Node v</b><br>Node: {v}"], hoverinfo="text",
            name="Query nodes", showlegend=False,
        )
    )

    xm, ym, zm = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    fig.add_trace(go.Scatter3d(x=[xm], y=[ym], z=[zm], mode="markers", marker=dict(size=7, color=COLORS["query_marker"], symbol="diamond", line=dict(width=2, color="black")), hovertext=[hover], hoverinfo="text", name="Query marker", showlegend=False))


def make_cora_pair_query_figure_3d(
    *,
    variant_key,
    full_cora_edges,
    train_edges,
    test_confusion_groups,
    num_nodes,
    u,
    v,
    score,
    threshold,
    testing_status_text,
    degree_map,
    neighbor_map,
    component_id_map,
    component_size_map,
    focus_view=False,
):
    full_cora_edges_tuple = tuple((int(a), int(b)) for a, b in full_cora_edges)
    pos = compute_full_cora_layout_3d(full_cora_edges_tuple, int(num_nodes))
    focus_nodes = get_focus_nodes(u=u, v=v, neighbor_map=neighbor_map, hops=1)
    fig = go.Figure()

    add_edge_trace_3d(fig, [(int(a), int(b)) for a, b in train_edges], pos, COLORS["train"], 0.9, "Training graph", False)

    focus_train_edges = filter_edges_touching_focus([(int(a), int(b)) for a, b in train_edges], focus_nodes)
    add_edge_trace_3d(fig, focus_train_edges, pos, COLORS["train_focus"], 1.8, "Training near query", False)

    for code in ["TN", "FP", "FN", "TP"]:
        items = test_confusion_groups.get(code, [])
        width = {"TP": 3.8, "FP": 3.4, "TN": 1.4, "FN": 4.0}[code]
        add_edge_trace_3d(fig, items, pos, COLORS[code], width, f"{code} ({len(items)})", True)

    add_nodes_trace_3d(fig, pos, num_nodes, u, v, degree_map, neighbor_map, component_id_map, component_size_map, focus_nodes)
    add_query_pair_dashed_trace_3d(fig, pos, u, v, score, threshold, testing_status_text)

    xaxis = dict(showbackground=False, showticklabels=False, title="", visible=False)
    yaxis = dict(showbackground=False, showticklabels=False, title="", visible=False)
    zaxis = dict(showbackground=False, showticklabels=False, title="", visible=False)

    camera = dict(eye=dict(x=1.55, y=1.55, z=1.20), center=dict(x=0, y=0, z=0))
    aspectmode = "data"

    if focus_view:
        ranges = get_focus_ranges_3d(pos, focus_nodes, padding_ratio=0.95, min_span=0.06)
        if ranges is not None:
            xr, yr, zr = ranges
            xaxis["range"] = xr
            yaxis["range"] = yr
            zaxis["range"] = zr
            camera = dict(eye=dict(x=0.95, y=0.95, z=0.75), center=dict(x=0, y=0, z=0))
            aspectmode = "cube"

    fig.update_layout(
        height=640,
        showlegend=True,
        margin=dict(l=0, r=0, t=12, b=52),
        scene=dict(
            xaxis=xaxis,
            yaxis=yaxis,
            zaxis=zaxis,
            camera=camera,
            aspectmode=aspectmode,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="rgba(0,0,0,0.12)",
            borderwidth=1,
            font=dict(size=9),
        ),
        uirevision=f"3d-pair-{variant_key}-{int(u)}-{int(v)}-{threshold}-{focus_view}",
    )
    return fig


# ============================================================
# DISPLAY HELPERS
# ============================================================
def metrics_dataframe(metrics):
    rows = []
    for variant_key, meta in VARIANTS.items():
        mk = meta["metrics_key"]
        values = metrics.get(mk, {})
        rows.append({"Variant": meta["title"], "AUC": values.get("auc", np.nan), "AP": values.get("ap", np.nan), "HR@100": values.get("hr100", np.nan)})
    return pd.DataFrame(rows)


def confusion_summary_dataframe(groups):
    return pd.DataFrame([{"Testing Status": CONFUSION_LABELS[code], "Code": code, "Count": len(groups.get(code, []))} for code in ["TP", "FP", "TN", "FN"]])


def render_flow_card(variant_key, u, v, score, pred_label, testing_code, testing_status_text, edge_status, score_key):
    display = VARIANTS[variant_key]
    status_color = STATUS_BADGE_COLORS[testing_code]
    prediction_text = "Link" if pred_label == 1 else "No-link"
    edge_status_text = "Link" if edge_status == 1 else "No-link"

    st.markdown(
        f"""
        <div class="comparison-card">
            <div class="flow-title">{display['title']}</div>
            <div class="flow-subtitle">{display['subtitle']}</div>
            <div class="pill-row">
                <span class="pill">{display['badge']}</span>
                <span class="status-pill" style="background:{status_color};">{testing_status_text}</span>
            </div>
            <div class="kv-grid">
                <div class="kv-key">score</div><div class="kv-val">{score:.4f}</div>
                <div class="kv-key">prediction</div><div class="kv-val">{prediction_text}</div>
                <div class="kv-key">Cora status</div><div class="kv-val">{edge_status_text}</div>
                <div class="kv-key">score key</div><div class="kv-val">{score_key}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_variant_flow(data, variant_key, u, v, full_cora_edge_set, test_edges, test_edges_false):
    score = get_prediction(data, variant_key, u, v)
    if score is None:
        raise ValueError(f"No score for {VARIANTS[variant_key]['short']} pair u={u}, v={v}")

    pred_label = 1 if score >= DEFAULT_THRESHOLD else 0
    testing_code, testing_status_text, edge_status = get_pair_testing_status(u, v, pred_label, full_cora_edge_set)
    test_groups = build_testing_confusion_edges(data, variant_key, DEFAULT_THRESHOLD, test_edges, test_edges_false)
    score_key = get_available_score_key(data, variant_key)

    return {
        "score": score,
        "pred_label": pred_label,
        "testing_code": testing_code,
        "testing_status_text": testing_status_text,
        "edge_status": edge_status,
        "test_groups": test_groups,
        "score_key": score_key,
    }


def reset_focus_flags():
    st.session_state["left_focus_view"] = False
    st.session_state["right_focus_view"] = False


# ============================================================
# MAIN APP
# ============================================================
inject_css()

st.title("GVFA Link Prediction Explorer")
st.markdown('<div class="small-caption">Single interface for selecting GVFA configuration, entering node pair, and comparing two variants.</div>', unsafe_allow_html=True)

phi_files = get_phi_files()
if len(phi_files) == 0:
    st.error("No visualization files found. Expected files like `visual_outputs/gvfa_visual_phi1.pkl`.")
    st.stop()

# ============================================================
# TOP CONFIGURATION AREA
# ============================================================
top_config_col, top_u_col, top_v_col = st.columns([1.4, 1, 1])

with top_config_col:
    selected_phi = st.selectbox("GVFA configuration", sorted(phi_files.keys()), key="selected_phi_top", on_change=reset_focus_flags)

# Load selected package after configuration selection
data = load_package(phi_files[selected_phi])
num_nodes = int(data["num_nodes"])

with top_u_col:
    u = st.number_input("Node u", min_value=0, max_value=num_nodes - 1, value=0, step=1, key="pair_u", on_change=reset_focus_flags)

with top_v_col:
    v = st.number_input("Node v", min_value=0, max_value=num_nodes - 1, value=1, step=1, key="pair_v", on_change=reset_focus_flags)

train_edges = data["train_edges"]
val_edges = data["val_edges"]
test_edges = data["test_edges"]
test_edges_false = data["test_edges_false"]

train_set = make_edge_set(train_edges)
val_set = make_edge_set(val_edges)
test_set = make_edge_set(test_edges)
false_set = make_edge_set(test_edges_false)

full_cora_edge_set = train_set | val_set | test_set
full_cora_edges = edge_set_to_sorted_list(full_cora_edge_set)
full_cora_edges_tuple = tuple((int(a), int(b)) for a, b in full_cora_edges)

degree_map, neighbor_map, component_id_map, component_size_map = compute_graph_metadata(full_cora_edges_tuple, num_nodes)

# ============================================================
# RESULTS SECTION
# ============================================================
st.subheader("Results")
metrics = data.get("metrics", {})
if metrics:
    st.dataframe(metrics_dataframe(metrics), use_container_width=True, hide_index=True)
else:
    st.info("No metrics found in the selected package.")

# ============================================================
# VARIANT SELECTION AREA
# ============================================================
st.subheader("Variant Comparison")

available = available_variant_options(data)
missing = [k for k in VARIANTS.keys() if k not in available]
if missing:
    st.warning("Missing score matrices for: " + ", ".join([VARIANTS[k]["short"] for k in missing]) + ". These variants require regenerated .pkl files with their score matrices.")

variant_col1, variant_col2, button_col = st.columns([1.35, 1.35, 0.8])
with variant_col1:
    left_variant = st.selectbox("Left variant", list(VARIANTS.keys()), index=0, format_func=lambda k: VARIANTS[k]["title"], key="left_variant", on_change=reset_focus_flags)
with variant_col2:
    right_variant = st.selectbox("Right variant", list(VARIANTS.keys()), index=1, format_func=lambda k: VARIANTS[k]["title"], key="right_variant", on_change=reset_focus_flags)
with button_col:
    st.write("")
    st.write("")
    compare_clicked = st.button("Compare", type="primary", use_container_width=True)

if "pair_query_active" not in st.session_state:
    st.session_state["pair_query_active"] = False
if "left_focus_view" not in st.session_state:
    st.session_state["left_focus_view"] = False
if "right_focus_view" not in st.session_state:
    st.session_state["right_focus_view"] = False

if compare_clicked:
    st.session_state["pair_query_active"] = True
    st.session_state["left_focus_view"] = False
    st.session_state["right_focus_view"] = False

# ============================================================
# COMPARISON OUTPUT
# ============================================================
if st.session_state["pair_query_active"]:
    u, v = int(u), int(v)

    if u == v:
        st.warning("Please select two different nodes.")
        st.stop()

    if left_variant == right_variant:
        st.warning("Please select two different variants to compare.")
        st.stop()

    unavailable = [k for k in [left_variant, right_variant] if not variant_is_available(data, k)]
    if unavailable:
        st.error("Selected variant score matrix is missing: " + ", ".join([VARIANTS[k]["title"] for k in unavailable]) + ".")
        st.stop()

    left_flow_data = run_variant_flow(data, left_variant, u, v, full_cora_edge_set, test_edges, test_edges_false)
    right_flow_data = run_variant_flow(data, right_variant, u, v, full_cora_edge_set, test_edges, test_edges_false)

    st.markdown(f"### Pair Query: u={u}, v={v}")

    left_flow, right_flow = st.columns(2, gap="large")

    with left_flow:
        render_flow_card(left_variant, u, v, left_flow_data["score"], left_flow_data["pred_label"], left_flow_data["testing_code"], left_flow_data["testing_status_text"], left_flow_data["edge_status"], left_flow_data["score_key"])

        title_col, focus_col, reset_col = st.columns([1.6, 0.85, 0.75])
        with title_col:
            st.markdown(f'<div class="graph-card-title">{VARIANTS[left_variant]["short"]} — 3D graph</div>', unsafe_allow_html=True)
        with focus_col:
            if st.button("Focus pair", key="left_focus_button", use_container_width=True):
                st.session_state["left_focus_view"] = True
        with reset_col:
            if st.button("Full view", key="left_full_button", use_container_width=True):
                st.session_state["left_focus_view"] = False

        fig_left = make_cora_pair_query_figure_3d(
            variant_key=left_variant,
            full_cora_edges=full_cora_edges,
            train_edges=train_edges,
            test_confusion_groups=left_flow_data["test_groups"],
            num_nodes=num_nodes,
            u=u,
            v=v,
            score=left_flow_data["score"],
            threshold=DEFAULT_THRESHOLD,
            testing_status_text=left_flow_data["testing_status_text"],
            degree_map=degree_map,
            neighbor_map=neighbor_map,
            component_id_map=component_id_map,
            component_size_map=component_size_map,
            focus_view=st.session_state["left_focus_view"],
        )
        st.plotly_chart(fig_left, use_container_width=True, config=PLOTLY_GRAPH_CONFIG)

    with right_flow:
        render_flow_card(right_variant, u, v, right_flow_data["score"], right_flow_data["pred_label"], right_flow_data["testing_code"], right_flow_data["testing_status_text"], right_flow_data["edge_status"], right_flow_data["score_key"])

        title_col, focus_col, reset_col = st.columns([1.6, 0.85, 0.75])
        with title_col:
            st.markdown(f'<div class="graph-card-title">{VARIANTS[right_variant]["short"]} — 3D graph</div>', unsafe_allow_html=True)
        with focus_col:
            if st.button("Focus pair", key="right_focus_button", use_container_width=True):
                st.session_state["right_focus_view"] = True
        with reset_col:
            if st.button("Full view", key="right_full_button", use_container_width=True):
                st.session_state["right_focus_view"] = False

        fig_right = make_cora_pair_query_figure_3d(
            variant_key=right_variant,
            full_cora_edges=full_cora_edges,
            train_edges=train_edges,
            test_confusion_groups=right_flow_data["test_groups"],
            num_nodes=num_nodes,
            u=u,
            v=v,
            score=right_flow_data["score"],
            threshold=DEFAULT_THRESHOLD,
            testing_status_text=right_flow_data["testing_status_text"],
            degree_map=degree_map,
            neighbor_map=neighbor_map,
            component_id_map=component_id_map,
            component_size_map=component_size_map,
            focus_view=st.session_state["right_focus_view"],
        )
        st.plotly_chart(fig_right, use_container_width=True, config=PLOTLY_GRAPH_CONFIG)

    st.subheader("Testing Status Summary")
    summary_left, summary_right = st.columns(2, gap="large")
    with summary_left:
        st.markdown(f"**{VARIANTS[left_variant]['short']}**")
        st.dataframe(confusion_summary_dataframe(left_flow_data["test_groups"]), use_container_width=True, hide_index=True)
    with summary_right:
        st.markdown(f"**{VARIANTS[right_variant]['short']}**")
        st.dataframe(confusion_summary_dataframe(right_flow_data["test_groups"]), use_container_width=True, hide_index=True)
