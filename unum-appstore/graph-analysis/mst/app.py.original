"""
MST Function - Computes Minimum Spanning Tree of the graph

Inspired by SeBS benchmark 502.graph-mst.
Uses Kruskal's algorithm with Union-Find.
"""
import datetime


class UnionFind:
    """Union-Find data structure for Kruskal's algorithm."""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def compute_mst(edge_list, num_nodes):
    """
    Compute Minimum Spanning Tree using Kruskal's algorithm.
    
    Args:
        edge_list: List of (u, v, weight) tuples
        num_nodes: Number of nodes
    
    Returns:
        dict with MST results
    """
    # Sort edges by weight
    sorted_edges = sorted(edge_list, key=lambda x: x[2])
    
    uf = UnionFind(num_nodes)
    mst_edges = []
    total_weight = 0
    
    for u, v, weight in sorted_edges:
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
            total_weight += weight
            
            # MST has n-1 edges
            if len(mst_edges) == num_nodes - 1:
                break
    
    # Check if graph is connected
    is_connected = len(mst_edges) == num_nodes - 1
    
    # Find min and max edge weights in MST
    if mst_edges:
        weights = [e[2] for e in mst_edges]
        min_weight = min(weights)
        max_weight = max(weights)
        avg_weight = sum(weights) / len(weights)
    else:
        min_weight = max_weight = avg_weight = 0
    
    return {
        "edges_in_mst": len(mst_edges),
        "total_weight": round(total_weight, 2),
        "is_connected": is_connected,
        "min_edge_weight": round(min_weight, 2),
        "max_edge_weight": round(max_weight, 2),
        "avg_edge_weight": round(avg_weight, 2)
    }


def lambda_handler(event, context):
    """
    Compute MST on the input graph.
    
    Input: Graph data from GraphGenerator
    Output: MST results
    """
    start_time = datetime.datetime.now()
    
    # Extract graph data
    graph = event.get("graph", event)
    edge_list = graph.get("edge_list", [])
    num_nodes = graph.get("nodes", 0)
    
    # Convert edge list if needed (from JSON arrays to tuples)
    if edge_list and isinstance(edge_list[0], list):
        edge_list = [tuple(e) for e in edge_list]
    
    # Compute MST
    result = compute_mst(edge_list, num_nodes)
    
    end_time = datetime.datetime.now()
    compute_time = (end_time - start_time) / datetime.timedelta(microseconds=1)
    
    return {
        "algorithm": "MST",
        "result": result,
        "compute_time_us": compute_time,
        "graph_nodes": num_nodes
    }
