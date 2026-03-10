"""
PageRank Function - Computes PageRank algorithm on the graph

Inspired by SeBS benchmark 501.graph-pagerank.
"""
from unum_streaming import StreamingPublisher, set_streaming_output
import datetime


def compute_pagerank(adj_list, num_nodes, damping=0.85, max_iterations=100, tolerance=1e-6):
    """
    Compute PageRank using power iteration method.
    
    Args:
        adj_list: Adjacency list representation of graph
        num_nodes: Number of nodes
        damping: Damping factor (typically 0.85)
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
    
    Returns:
        dict with PageRank scores and metadata
    """
    # Initialize PageRank scores uniformly
    pagerank = {i: 1.0 / num_nodes for i in range(num_nodes)}
    
    # Compute out-degrees
    out_degree = {i: len(neighbors) for i, neighbors in adj_list.items()}
    
    # Build reverse adjacency list (incoming edges)
    in_neighbors = {i: [] for i in range(num_nodes)}
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            in_neighbors[neighbor].append(node)
    
    iterations = 0
    for iteration in range(max_iterations):
        new_pagerank = {}
        diff = 0
        
        for node in range(num_nodes):
            # Sum of PR contributions from incoming neighbors
            rank_sum = 0
            for in_node in in_neighbors[node]:
                if out_degree[in_node] > 0:
                    rank_sum += pagerank[in_node] / out_degree[in_node]
            
            # Apply damping factor
            new_pagerank[node] = (1 - damping) / num_nodes + damping * rank_sum
            diff += abs(new_pagerank[node] - pagerank[node])
        
        pagerank = new_pagerank
        iterations = iteration + 1
        
        if diff < tolerance:
            break
    
    # Find top nodes
    sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, score in sorted_nodes[:5]]
    
    return {
        "scores": {str(k): v for k, v in sorted_nodes[:10]},  # Top 10 scores
        "top_nodes": top_nodes,
        "iterations": iterations,
        "converged": iterations < max_iterations
    }


def lambda_handler(event, context):
    """
    Compute PageRank on the input graph.
    
    Input: Graph data from GraphGenerator
    Output: PageRank results
    """

    # Streaming: Initialize publisher for incremental parameter streaming
    _streaming_session = (event.get('Session', '') if isinstance(event, dict) else '') or str(id(event))
    _streaming_publisher = StreamingPublisher(
        session_id=_streaming_session,
        source_function="PageRankFunction",
        field_names=["graph_nodes", "result", "compute_time_us"]
    )
    start_time = datetime.datetime.now()
    
    # Extract graph data
    graph = event.get("graph", event)
    adj_list = graph.get("adjacency_list", {})
    num_nodes = graph.get("nodes", len(adj_list))
    _streaming_publisher.publish('graph_nodes', num_nodes)
    # Streaming: Signal to runtime to invoke next function early with futures
    if _streaming_publisher.should_invoke_next():
        _streaming_payload = _streaming_publisher.get_streaming_payload()
        # Store payload for runtime to pick up and invoke continuation
        set_streaming_output(_streaming_payload)
        _streaming_publisher.mark_next_invoked()
    
    # Convert string keys back to integers if needed
    if adj_list and isinstance(list(adj_list.keys())[0], str):
        adj_list = {int(k): v for k, v in adj_list.items()}
    
    # Compute PageRank
    result = compute_pagerank(adj_list, num_nodes)
    _streaming_publisher.publish('result', result)
    
    end_time = datetime.datetime.now()
    compute_time = (end_time - start_time) / datetime.timedelta(microseconds=1)
    _streaming_publisher.publish('compute_time_us', compute_time)
    
    return {
        "algorithm": "PageRank",
        "result": result,
        "compute_time_us": compute_time,
        "graph_nodes": num_nodes
    }
