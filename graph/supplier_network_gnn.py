"""
Supplier Network Graph Neural Network (GraphSAGE)
==================================================
Models supply chain contagion risk — how a disruption at one supplier
propagates through the dependency graph to second and third-tier suppliers.

Architecture: GraphSAGE with mean aggregation (Hamilton et al., 2017)
  - Node features: ML risk score, on-time rate, financial health, geo risk,
    spend share, network centrality measures
  - Edge features: material flow volume, sole-source flag, geographic proximity
  - Task: predict P(node j disrupted | node i disrupted) for all (i,j) pairs

Reference:
  Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive Representation
  Learning on Large Graphs. NeurIPS.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


NODE_FEATURES = [
    "disruption_probability",
    "on_time_rate_90d",
    "financial_risk_score",
    "geo_risk_score",
    "spend_share_pct",
    "lead_time_variance_ratio",
    "network_pagerank",           # will be computed iteratively
    "betweenness_centrality",     # from NetworkX pre-processing
    "sole_source_flag",           # 1 if only supplier for ≥1 SKU
]


# ── GraphSAGE Model ────────────────────────────────────────────────────────
class SupplierGraphSAGE(nn.Module):
    """
    Two-layer GraphSAGE with skip connections and edge-conditioned aggregation.
    Outputs node embeddings used for:
      (a) Node-level: disruption risk refinement (incorporating network context)
      (b) Edge-level: pairwise contagion probability P(j disrupted | i disrupted)
    """
    def __init__(
        self,
        in_channels:   int,
        hidden_dim:    int = 64,
        out_dim:       int = 32,
        n_layers:      int = 2,
        dropout:       float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        dims = [in_channels] + [hidden_dim] * (n_layers - 1) + [out_dim]
        for i in range(n_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1], aggr="mean"))
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))

        # Edge classifier: predicts P(contagion) from concatenated node embeddings
        self.edge_clf = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Node-level disruption risk refiner
        self.node_clf = nn.Sequential(
            nn.Linear(out_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Graph convolution layers with residual connection
        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h_new = conv(h, edge_index)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            if h.shape == h_new.shape:
                h = h + h_new               # skip connection
            else:
                h = h_new

        # Node risk scores
        node_risk = self.node_clf(h).squeeze(-1)   # (N,)

        return h, node_risk

    def predict_contagion(self, embeddings: torch.Tensor,
                          edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict P(j disrupted | i disrupted) for every edge (i→j).
        Uses concatenated source and destination embeddings.
        """
        src, dst = edge_index
        edge_feats = torch.cat([embeddings[src], embeddings[dst]], dim=-1)
        return self.edge_clf(edge_feats).squeeze(-1)


# ── Training pipeline ──────────────────────────────────────────────────────
class SupplierNetworkModel:
    def __init__(
        self,
        hidden_dim:  int   = 64,
        out_dim:     int   = 32,
        n_layers:    int   = 2,
        lr:          float = 5e-4,
        epochs:      int   = 100,
        dropout:     float = 0.2,
    ):
        self.hidden_dim = hidden_dim
        self.out_dim    = out_dim
        self.n_layers   = n_layers
        self.lr         = lr
        self.epochs     = epochs
        self.dropout    = dropout
        self.model      = None

    def _build_graph(
        self,
        suppliers:     pd.DataFrame,
        edges:         pd.DataFrame,
    ) -> Data:
        """
        Build PyG Data object from supplier node features and edge list.

        edges: DataFrame with columns [src_supplier_id, dst_supplier_id,
                                        flow_volume, sole_source_edge, contagion_label]
        """
        sup_idx = {s: i for i, s in enumerate(suppliers["supplier_id"])}

        # Node feature matrix
        x = torch.tensor(
            suppliers[NODE_FEATURES].fillna(0).values.astype(np.float32),
            dtype=torch.float32,
        )

        # Edge index (directed)
        src = edges["src_supplier_id"].map(sup_idx).values
        dst = edges["dst_supplier_id"].map(sup_idx).values
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(suppliers))

        # Edge labels for contagion prediction
        y_edge = torch.tensor(
            edges["contagion_label"].values.astype(np.float32),
            dtype=torch.float32,
        ) if "contagion_label" in edges.columns else None

        # Node labels (ground truth disruption)
        y_node = torch.tensor(
            suppliers["disruption_label"].values.astype(np.float32),
            dtype=torch.float32,
        ) if "disruption_label" in suppliers.columns else None

        return Data(x=x, edge_index=edge_index, y_edge=y_edge, y_node=y_node)

    def train(
        self,
        suppliers: pd.DataFrame,
        edges:     pd.DataFrame,
    ) -> dict:
        graph = self._build_graph(suppliers, edges).to(DEVICE)
        n_features = len(NODE_FEATURES)

        self.model = SupplierGraphSAGE(
            in_channels=n_features,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(DEVICE)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                     weight_decay=1e-4)
        criterion_node = nn.BCELoss()
        criterion_edge = nn.BCELoss()

        print(f"[SupplierGNN] Training on {len(suppliers)} nodes, "
              f"{len(edges)} edges for {self.epochs} epochs...")

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            optimiser.zero_grad()
            embeddings, node_risk = self.model(graph.x, graph.edge_index)

            loss = torch.tensor(0.0, device=DEVICE)

            # Node disruption loss
            if graph.y_node is not None:
                n_edges_added = graph.edge_index.shape[1] - len(edges)
                n_original    = len(suppliers)
                loss += criterion_node(node_risk[:n_original], graph.y_node)

            # Edge contagion loss
            if graph.y_edge is not None:
                n_orig_edges = len(edges)
                orig_ei = graph.edge_index[:, :n_orig_edges]
                contagion_prob = self.model.predict_contagion(embeddings, orig_ei)
                loss += criterion_edge(contagion_prob, graph.y_edge.to(DEVICE))

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimiser.step()

            if epoch % 20 == 0:
                print(f"  Epoch {epoch:03d} | Loss: {loss.item():.5f}")

        return {"epochs": self.epochs, "final_loss": loss.item()}

    def get_contagion_matrix(
        self,
        suppliers: pd.DataFrame,
        edges:     pd.DataFrame,
    ) -> np.ndarray:
        """
        Returns (N, N) contagion probability matrix for all supplier pairs.
        Used as input to Monte Carlo VaR simulator.
        """
        graph = self._build_graph(suppliers, edges).to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            embeddings, node_risk = self.model(graph.x, graph.edge_index)

        n = len(suppliers)
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    pair = torch.tensor([[i], [j]], dtype=torch.long).to(DEVICE)
                    C[i, j] = self.model.predict_contagion(
                        embeddings, pair
                    ).item()
        return C

    def get_node_embeddings(
        self,
        suppliers: pd.DataFrame,
        edges:     pd.DataFrame,
    ) -> np.ndarray:
        graph = self._build_graph(suppliers, edges).to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            embeddings, _ = self.model(graph.x, graph.edge_index)
        return embeddings.cpu().numpy()


# ── Standalone demo ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import torch_geometric
    except ImportError:
        print("torch_geometric not installed. Run: pip install torch-geometric")
        print("Showing architecture summary only.\n")
        model = SupplierGraphSAGE(in_channels=len(NODE_FEATURES))
        total = sum(p.numel() for p in model.parameters())
        print(f"GraphSAGE architecture: {model}")
        print(f"\nTotal parameters: {total:,}")
        exit()

    np.random.seed(42)
    N_SUP, N_EDGES = 25, 60

    suppliers = pd.DataFrame({
        "supplier_id":               [f"SUP-{i:02d}" for i in range(N_SUP)],
        "disruption_probability":    np.random.uniform(0.05, 0.70, N_SUP),
        "on_time_rate_90d":          np.random.beta(8, 2, N_SUP),
        "financial_risk_score":      np.random.beta(3, 7, N_SUP),
        "geo_risk_score":            np.random.uniform(0, 1, N_SUP),
        "spend_share_pct":           np.random.dirichlet(np.ones(N_SUP)) * 100,
        "lead_time_variance_ratio":  np.random.exponential(0.3, N_SUP),
        "network_pagerank":          np.random.exponential(0.05, N_SUP),
        "betweenness_centrality":    np.random.exponential(0.02, N_SUP),
        "sole_source_flag":          np.random.binomial(1, 0.2, N_SUP).astype(float),
        "disruption_label":          np.random.binomial(1, 0.2, N_SUP).astype(float),
    })

    srcs = np.random.randint(0, N_SUP, N_EDGES)
    dsts = np.random.randint(0, N_SUP, N_EDGES)
    edges = pd.DataFrame({
        "src_supplier_id": [f"SUP-{s:02d}" for s in srcs],
        "dst_supplier_id": [f"SUP-{d:02d}" for d in dsts],
        "flow_volume":     np.random.exponential(100_000, N_EDGES),
        "sole_source_edge":np.random.binomial(1, 0.15, N_EDGES).astype(float),
        "contagion_label": np.random.binomial(1, 0.25, N_EDGES).astype(float),
    })

    gnn = SupplierNetworkModel(hidden_dim=32, out_dim=16, epochs=40)
    gnn.train(suppliers, edges)

    C = gnn.get_contagion_matrix(suppliers, edges)
    print(f"\nContagion matrix shape: {C.shape}")
    print(f"Mean contagion probability: {C.mean():.3f}")
    print(f"Max contagion probability:  {C.max():.3f}")

    emb = gnn.get_node_embeddings(suppliers, edges)
    print(f"Node embedding shape: {emb.shape}")
    print("\nTop 5 suppliers by average outbound contagion:")
    avg_contagion = C.mean(axis=1)
    top5 = np.argsort(avg_contagion)[-5:][::-1]
    for i in top5:
        print(f"  {suppliers.iloc[i]['supplier_id']} | "
              f"avg contagion: {avg_contagion[i]:.3f} | "
              f"risk: {suppliers.iloc[i]['disruption_probability']:.3f}")
