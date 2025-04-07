# connectogram_plot.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
# ===============================
# Config: Set your Connecto path
# ===============================
PATH_Data = os.path.join(Path(__file__).resolve().parent.parent, 'Data')

def load_data(path_connecto):
    """Load label and graph data from CSV files."""
    labels = pd.read_csv(os.path.join(PATH_Data, 'data_atlas.csv'))
    df_graph = pd.read_csv(os.path.join(PATH_Data, 'data_con_figures.csv'))
    return labels, df_graph


def preprocess_graph_data(df):
    """Apply filtering steps to the input DataFrame."""
    df = df[(df.Num_trial > 10) & (df.Hemi != 'B') & (df.Sig > 0)]
    df = df[np.isin(df.Group, ['direct', 'indirect'])]
    df = df[df.StimR != df.ChanR]
    return df.reset_index(drop=True)


def create_region_color_map(labels):
    """Return a mapping from region to plot_color."""
    return labels.drop_duplicates('region').set_index('region')['plot_color'].to_dict()


def normalize_values(values, min_value=0.5, max_value=2):
    """Normalize values to a given range [min_value, max_value]."""
    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        return [min_value] * len(values)
    return [min_value + (val - min_val) / (max_val - min_val) * (max_value - min_value) for val in values]


def build_graph(df):
    """Build a NetworkX graph with nodes and weighted edges."""
    G = nx.Graph()
    mean_sig = df.groupby("StimR")["Sig"].mean()

    for region, size in mean_sig.items():
        G.add_node(region, size=size)

    for stimr in df['StimR'].unique():
        for chanr in df['ChanR'].unique():
            if stimr == chanr:
                continue
            df_sel = df[((df['StimR'] == stimr) & (df['ChanR'] == chanr)) |
                        ((df['StimR'] == chanr) & (df['ChanR'] == stimr))]
            if not df_sel.empty:
                sig_mean = df_sel['Sig'].mean()
                latency_mean = df_sel['peak_latency'].mean()
                if sig_mean > 0.4:
                    G.add_edge(stimr, chanr,
                               weight=sig_mean,
                               latency=1 / latency_mean,
                               raw_latency=latency_mean)
    return G


def draw_graph(G, region_color_map):
    """Visualize the graph with normalized node sizes and edge weights."""
    pos = nx.spring_layout(G, weight='latency', k=0.05, iterations=50, seed=41)

    node_sizes_raw = [G.nodes[n]["size"] for n in G.nodes()]
    node_sizes = np.array(normalize_values(node_sizes_raw)) * 500
    node_colors = [region_color_map.get(n, "#CCCCCC") for n in G.nodes()]  # fallback to grey

    edge_weights_raw = [edata['weight'] for _, _, edata in G.edges(data=True)]
    edge_weights = np.array(normalize_values(edge_weights_raw))

    low_latency_edges = [(u, v) for (u, v, edata) in G.edges(data=True) if edata['raw_latency'] <= 0.065]
    high_latency_edges = [(u, v) for (u, v, edata) in G.edges(data=True) if edata['raw_latency'] > 0.065]

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.7)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color='black')

    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=low_latency_edges, width=edge_weights, edge_color='gray', style='-', alpha=0.7)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=high_latency_edges, width=edge_weights, edge_color='gray', style='--', alpha=0.7)

    ax.set_title("Connectogram Network", fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    print('Figure 1i')


def main():
    labels, df_graph = load_data(PATH_CONNECTO)
    df_graph = preprocess_graph_data(df_graph)
    region_color_map = create_region_color_map(labels)
    G = build_graph(df_graph)
    draw_graph(G, region_color_map)


if __name__ == '__main__':
    main()
