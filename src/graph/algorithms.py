"""
ClaimGraph AI — Graph Algorithms (NetworkX)
=============================================
Ye module claims ka graph banata hai aur graph algorithms chalata hai.

Why NetworkX (not Neo4j GDS)?
- No plugin/license needed — pure Python mein sab hota hai
- Student easily explain kar sakta hai har algorithm step by step
- 25k claims ke liye performance sufficient hai
- Debug karna easy hai — breakpoints laga sakte ho

Algorithms implemented:
1. Degree centrality — kitne entities ek node touch karta hai
2. Louvain community detection — fraud rings as dense communities
3. Connected components — isolated suspicious clusters
4. PageRank — unusually "central" entities find karta hai
"""

import pandas as pd
import numpy as np
import networkx as nx
from community import community_louvain  # python-louvain package
from collections import defaultdict
from typing import Dict, Tuple
import json
import os


class ClaimGraphAnalyzer:
    """
    Claims ka graph banao aur analyze karo.

    Graph structure:
    - Nodes: claims, customers, devices, addresses, repair_shops, payment_accounts
    - Edges: relationships (FILED, USES_DEVICE, LOCATED_AT, REPAIRED_BY, PAID_WITH)

    Interview tip: "I build a heterogeneous graph where claims connect to shared
    entities. When multiple claims share the same device or repair shop, they become
    neighbors in the graph. Community detection then finds clusters of unusually
    connected claims — which are exactly the fraud rings I injected."
    """

    def __init__(self):
        self.graph = nx.Graph()
        self.claim_nodes = set()  # Track which nodes are claims

    def build_graph(self, claims_df: pd.DataFrame,
                    customers_df: pd.DataFrame,
                    devices_df: pd.DataFrame,
                    repair_shops_df: pd.DataFrame,
                    addresses_df: pd.DataFrame,
                    payment_accounts_df: pd.DataFrame) -> nx.Graph:
        """
        Heterogeneous graph banao — har entity type ek node,
        relationships edges hain.

        Ye graph ka structure interview mein explain karna important hai:
        "A claim connects to its customer, device, repair shop, and address.
        When two claims share the same device, they're connected through that
        device node. This is how the graph reveals hidden connections that
        aren't visible in a flat table."
        """
        print("🕸️  Building claims graph...")

        # === Add Claim nodes ===
        for _, row in claims_df.iterrows():
            claim_id = row["claim_id"]
            self.graph.add_node(claim_id, node_type="claim",
                              claim_amount=row.get("claim_amount", 0),
                              claim_type=row.get("claim_type", "unknown"),
                              claim_date=str(row.get("claim_date", "")))
            self.claim_nodes.add(claim_id)

        # === Add entity nodes and edges ===

        # Customer nodes + FILED edges
        for _, row in claims_df.iterrows():
            cust_id = row["customer_id"]
            self.graph.add_node(cust_id, node_type="customer")
            self.graph.add_edge(row["claim_id"], cust_id,
                              edge_type="FILED_BY")

        # Device nodes + USES_DEVICE edges
        for _, row in claims_df.iterrows():
            dev_id = row["device_id"]
            self.graph.add_node(dev_id, node_type="device")
            self.graph.add_edge(row["claim_id"], dev_id,
                              edge_type="USES_DEVICE")

        # Repair shop nodes + REPAIRED_BY edges
        for _, row in claims_df.iterrows():
            if pd.notna(row.get("repair_shop_id")):
                shop_id = row["repair_shop_id"]
                self.graph.add_node(shop_id, node_type="repair_shop")
                self.graph.add_edge(row["claim_id"], shop_id,
                                  edge_type="REPAIRED_BY")

        # Customer → Address edges (LOCATED_AT)
        for _, row in customers_df.iterrows():
            cust_id = row["customer_id"]
            addr_id = row["address_id"]
            self.graph.add_node(addr_id, node_type="address")
            if self.graph.has_node(cust_id):
                self.graph.add_edge(cust_id, addr_id,
                                  edge_type="LOCATED_AT")

        # Customer → Payment Account edges (PAID_WITH)
        for _, row in customers_df.iterrows():
            cust_id = row["customer_id"]
            pay_id = row["payment_account_id"]
            self.graph.add_node(pay_id, node_type="payment_account")
            if self.graph.has_node(cust_id):
                self.graph.add_edge(cust_id, pay_id,
                                  edge_type="PAID_WITH")

        print(f"   ✅ Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        print(f"   📊 Claim nodes: {len(self.claim_nodes)}")

        return self.graph

    def compute_degree_centrality(self) -> Dict[str, int]:
        """
        Degree centrality — har node kitne doosre nodes se connected hai.

        Fraud signal: Agar ek device node ka degree bahut high hai,
        matlab bahut saare claims us device ko reference kar rahe hain —
        ye shared-device fraud ring ho sakta hai.

        Simple calculation: degree = number of edges connected to node.
        """
        print("📊 Computing degree centrality...")
        degrees = dict(self.graph.degree())
        claim_degrees = {n: degrees[n] for n in self.claim_nodes if n in degrees}
        avg_degree = np.mean(list(claim_degrees.values())) if claim_degrees else 0
        print(f"   ✅ Avg claim degree: {avg_degree:.2f}")
        return claim_degrees

    def compute_louvain_communities(self) -> Dict[str, int]:
        """
        Louvain community detection — graph ko dense communities mein split karo.

        WHY THIS IS THE CORE: Fraud rings dense communities banate hain kyunki
        ring members shared entities (devices, shops, payments) ke through
        tightly connected hote hain. Legitimate claims loosely connected hote hain.

        Interview answer: "Louvain maximizes modularity — it finds groups of nodes
        that are more connected to each other than to the rest of the graph.
        Fraud rings naturally form such groups because ring members share
        devices, repair shops, or payment accounts."

        Returns: {node_id: community_id}
        """
        print("📊 Computing Louvain communities...")
        partition = community_louvain.best_partition(self.graph, random_state=42)

        # Community sizes calculate karo
        community_sizes = defaultdict(int)
        for node, comm_id in partition.items():
            community_sizes[comm_id] += 1

        n_communities = len(community_sizes)
        avg_size = np.mean(list(community_sizes.values()))
        print(f"   ✅ Found {n_communities} communities, avg size: {avg_size:.1f}")

        # Store community sizes for feature engineering
        self._community_sizes = community_sizes
        self._partition = partition

        return partition

    def compute_connected_components(self) -> Dict[str, int]:
        """
        Connected components — disjoint subgraphs find karo.

        Isolated clusters jo main graph se disconnected hain — ye suspicious
        hote hain kyunki legitimate claims usually ek bada connected component
        banate hain (through shared cities, common repair shops, etc.)

        Returns: {node_id: component_id}
        """
        print("📊 Computing connected components...")
        components = {}
        for i, component in enumerate(nx.connected_components(self.graph)):
            for node in component:
                components[node] = i

        n_components = len(set(components.values()))
        print(f"   ✅ Found {n_components} connected components")
        return components

    def compute_pagerank(self, alpha: float = 0.85) -> Dict[str, float]:
        """
        PageRank — unusually "central" entities find karo.

        PageRank originally web pages rank karne ke liye tha, lekin yahan
        ye batata hai ki kaun se nodes graph mein sabse influential hain.

        Fraud signal: Ek repair shop jiska PageRank bahut high hai
        (compared to other shops) = wo unusually many claims se connected hai.

        Interview answer: "PageRank tells me which entities are unusually central
        in the claims network. A repair shop with high PageRank touches more
        claims than expected, which could indicate it's part of a fraud ring
        inflating claims."
        """
        print("📊 Computing PageRank...")
        pagerank = nx.pagerank(self.graph, alpha=alpha)
        claim_pagerank = {n: pagerank[n] for n in self.claim_nodes if n in pagerank}
        avg_pr = np.mean(list(claim_pagerank.values())) if claim_pagerank else 0
        print(f"   ✅ Avg claim PageRank: {avg_pr:.6f}")
        return pagerank

    def compute_shared_entity_counts(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Har claim ke liye count karo ki kitne doosre claims same entities share karte hain.

        Ye DIRECT fraud features hain:
        - n_shared_devices: kitne claims same device use karte hain
        - n_shared_addresses: kitne claims same address se hain
        - n_shared_shops: kitne claims same repair shop use karte hain
        - n_shared_payments: kitne claims same payment account use karte hain

        Interview tip: "These are hand-engineered graph features. Each one
        counts how many other claims share the same entity with this claim.
        A claim that shares a device with 5 other claims is more suspicious
        than one that shares it with 0."
        """
        print("📊 Computing shared entity counts...")

        # Device sharing
        device_counts = claims_df.groupby("device_id")["claim_id"].transform("count")
        claims_df = claims_df.copy()
        claims_df["n_shared_devices"] = device_counts - 1  # Minus self

        # Repair shop sharing
        shop_claims = claims_df[claims_df["repair_shop_id"].notna()]
        shop_counts = shop_claims.groupby("repair_shop_id")["claim_id"].transform("count")
        claims_df["n_shared_shops"] = 0
        claims_df.loc[shop_claims.index, "n_shared_shops"] = shop_counts - 1

        # Address sharing (through customer)
        if "address_id" in claims_df.columns:
            addr_counts = claims_df.groupby("address_id")["claim_id"].transform("count")
            claims_df["n_shared_addresses"] = addr_counts - 1
        else:
            claims_df["n_shared_addresses"] = 0

        # Payment sharing (through customer)
        if "payment_account_id" in claims_df.columns:
            pay_counts = claims_df.groupby("payment_account_id")["claim_id"].transform("count")
            claims_df["n_shared_payments"] = pay_counts - 1
        else:
            claims_df["n_shared_payments"] = 0

        print(f"   ✅ Shared entity counts computed for {len(claims_df)} claims")
        return claims_df

    def extract_claim_features(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Saare graph algorithms ko combine karke per-claim features banao.
        Ye features ML model mein jaate hain.

        Output columns:
        - node_degree, community_id, community_size, pagerank_score
        - n_shared_devices, n_shared_addresses, n_shared_shops, n_shared_payments
        """
        print("\n🔧 Extracting graph features for all claims...")

        # Run all algorithms
        degrees = self.compute_degree_centrality()
        communities = self.compute_louvain_communities()
        pagerank = self.compute_pagerank()

        # Shared entity counts
        claims_with_sharing = self.compute_shared_entity_counts(claims_df)

        # Compile features
        features = []
        for claim_id in claims_df["claim_id"]:
            community_id = communities.get(claim_id, -1)
            features.append({
                "claim_id": claim_id,
                "node_degree": degrees.get(claim_id, 0),
                "community_id": community_id,
                "community_size": self._community_sizes.get(community_id, 0),
                "pagerank_score": pagerank.get(claim_id, 0),
            })

        features_df = pd.DataFrame(features)

        # Merge shared entity counts
        sharing_cols = ["claim_id", "n_shared_devices", "n_shared_addresses",
                       "n_shared_shops", "n_shared_payments"]
        available_cols = [c for c in sharing_cols if c in claims_with_sharing.columns]
        features_df = features_df.merge(
            claims_with_sharing[available_cols],
            on="claim_id", how="left"
        )

        print(f"   ✅ Graph features extracted: {len(features_df)} claims, "
              f"{len(features_df.columns)-1} features")

        return features_df

    def get_claim_neighborhood(self, claim_id: str, hops: int = 2) -> dict:
        """
        N-hop neighborhood of a claim — investigation dashboard ke liye.
        Returns nodes and edges for visualization.
        """
        if claim_id not in self.graph:
            return {"nodes": [], "edges": [], "error": f"Claim {claim_id} not found"}

        # BFS for N hops
        visited = {claim_id}
        frontier = {claim_id}
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier

        # Build subgraph
        subgraph = self.graph.subgraph(visited)
        nodes = []
        for node in subgraph.nodes():
            node_data = self.graph.nodes[node]
            nodes.append({
                "id": node,
                "type": node_data.get("node_type", "unknown"),
                **{k: v for k, v in node_data.items() if k != "node_type"}
            })

        edges = []
        for u, v, data in subgraph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "type": data.get("edge_type", "CONNECTED")
            })

        return {"nodes": nodes, "edges": edges}

    def detect_fraud_rings(self, min_community_size: int = 3,
                           min_shared_entities: int = 2) -> pd.DataFrame:
        """
        Detected fraud rings return karo — communities jo suspiciously dense hain.

        Logic: Ek community suspicious hai agar:
        1. Minimum 3 claim nodes hain
        2. Community mein shared entities (devices/shops/payments) >= 2

        Returns DataFrame with ring_id, member claims, ring_type, confidence.
        """
        print("🔍 Detecting fraud rings from graph communities...")

        if not hasattr(self, '_partition'):
            self.compute_louvain_communities()

        # Group claims by community
        community_claims = defaultdict(list)
        for node, comm_id in self._partition.items():
            if node in self.claim_nodes:
                community_claims[comm_id].append(node)

        rings = []
        ring_counter = 0
        for comm_id, claims in community_claims.items():
            if len(claims) >= min_community_size:
                # Check density — suspicious communities have more internal edges
                subgraph = self.graph.subgraph(
                    [n for n, c in self._partition.items() if c == comm_id]
                )
                density = nx.density(subgraph)

                # High density community with multiple claims = potential ring
                if density > 0.1 or len(claims) >= 4:
                    ring_counter += 1
                    for claim_id in claims:
                        rings.append({
                            "detected_ring_id": f"DRING-{ring_counter:03d}",
                            "claim_id": claim_id,
                            "community_id": comm_id,
                            "community_size": self._community_sizes[comm_id],
                            "community_density": round(density, 4),
                            "n_claim_members": len(claims)
                        })

        rings_df = pd.DataFrame(rings)
        n_rings = rings_df["detected_ring_id"].nunique() if len(rings_df) > 0 else 0
        print(f"   ✅ Detected {n_rings} suspicious rings")
        return rings_df

    def save_graph_data(self, output_dir: str):
        """Graph data save karo — dashboard aur later analysis ke liye"""
        os.makedirs(output_dir, exist_ok=True)

        # Node list
        nodes = []
        for node, data in self.graph.nodes(data=True):
            nodes.append({"id": node, **data})
        pd.DataFrame(nodes).to_csv(os.path.join(output_dir, "graph_nodes.csv"), index=False)

        # Edge list
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({"source": u, "target": v, **data})
        pd.DataFrame(edges).to_csv(os.path.join(output_dir, "graph_edges.csv"), index=False)

        print(f"   ✅ Graph data saved to {output_dir}/")
