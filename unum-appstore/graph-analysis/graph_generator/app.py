"""
Graph Generator Function - Entry point for the Graph Analysis Pipeline

Inspired by SeBS (Serverless Benchmark Suite) graph benchmarks.
Generates a random Barabási-Albert scale-free graph and prepares it for parallel processing.
"""
import datetime
import json
from unum_streaming import StreamingPublisher, set_streaming_output
import random


def generate_graph(size, seed=None):
    """
    Generate a Barabási-Albert scale-free graph.
    
    This is a simplified implementation that doesn't require igraph,
    making it easier to deploy on Lambda without heavy dependencies.
    
    Args:
        size: Number of nodes in the graph
        seed: Random seed for reproducibility
    
    Returns:
        dict with nodes, edges, and adjacency list
    """
    if seed is not None:
        random.seed(seed)
    
    # Start with a small complete graph of m0 nodes
    m = 3  # Number of edges to attach from a new node
    m0 = m + 1  # Initial nodes
    
    # Initialize adjacency list and degree list
    adj_list = {i: list(range(m0)) for i in range(m0)}
    for i in range(m0):
        adj_list[i] = [j for j in range(m0) if j != i]
    
    degrees = [m0 - 1] * m0  # Initial degree is m0-1 for complete graph
    edges = []
    
    # Add edges from initial complete graph
    for i in range(m0):
        for j in range(i + 1, m0):
            edges.append((i, j, random.uniform(1, 10)))  # weighted edges
    
    # Add remaining nodes with preferential attachment
    for new_node in range(m0, size):
        adj_list[new_node] = []
        
        # Calculate probabilities based on degree
        total_degree = sum(degrees)
        probabilities = [d / total_degree for d in degrees]
        
        # Select m nodes to connect to (preferential attachment)
        targets = set()
        while len(targets) < m:
            # Weighted random selection
            r = random.random()
            cumulative = 0
            for node, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    if node not in targets:
                        targets.add(node)
                    break
        
        # Add edges
        for target in targets:
            adj_list[new_node].append(target)
            adj_list[target].append(new_node)
            edges.append((new_node, target, random.uniform(1, 10)))
            degrees[target] += 1
        
        degrees.append(len(targets))
    
    return {
        "nodes": size,
        "edges": len(edges),
        "adjacency_list": adj_list,
        "edge_list": edges,
        "m": m
    }


def lambda_handler(event, context):
    """
    Generate a graph and return it for parallel processing.
    
    Input: {"size": 500, "seed": 42}
    Output: Graph data structure for downstream functions
    """
    # Default values

    # Streaming: Initialize publisher for incremental parameter streaming
    _streaming_session = (event.get('Session', '') if isinstance(event, dict) else '') or str(id(event))
    _streaming_publisher = StreamingPublisher(
        session_id=_streaming_session,
        source_function="GraphGeneratorFunction",
        field_names=["graph", "metadata"]
    )
    size = 100
    seed = None
    
    # Parse input
    if isinstance(event, dict):
        size = event.get("size", 100)
        seed = event.get("seed", None)
    
    # Limit size to prevent Lambda timeout
    size = min(size, 1000)
    
    start_time = datetime.datetime.now()
    
    # Generate the graph
    graph = generate_graph(size, seed)
    _streaming_publisher.publish('graph', graph)
    # Streaming: Signal to runtime to invoke next function early with futures
    if _streaming_publisher.should_invoke_next():
        _streaming_payload = _streaming_publisher.get_streaming_payload()
        # Store payload for runtime to pick up and invoke continuation
        set_streaming_output(_streaming_payload)
        _streaming_publisher.mark_next_invoked()
    
    end_time = datetime.datetime.now()
    generation_time = (end_time - start_time) / datetime.timedelta(microseconds=1)
    _stream_metadata = {'generation_time_us': generation_time, 'requested_size': size, 'seed': seed}
    _streaming_publisher.publish('metadata', _stream_metadata)
    
    # Return graph data for parallel processing
    result = {
        "graph": graph,
        "metadata": {
            "generation_time_us": generation_time,
            "requested_size": size,
            "seed": seed
        }
    }
    
    return result
