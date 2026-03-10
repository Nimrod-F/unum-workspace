"""
BFS Function - Performs Breadth-First Search on the graph

Inspired by SeBS benchmark 503.graph-bfs.
"""
import datetime
from unum_streaming import StreamingPublisher, set_streaming_output
from collections import deque


def compute_bfs(adj_list, num_nodes, source=0):
    """
    Perform Breadth-First Search from a source node.
    
    Args:
        adj_list: Adjacency list representation of graph
        num_nodes: Number of nodes
        source: Starting node for BFS
    
    Returns:
        dict with BFS results and statistics
    """
    visited = set()
    distances = {source: 0}
    parent = {source: None}
    queue = deque([source])
    visited.add(source)
    
    max_depth = 0
    nodes_per_level = {0: 1}
    
    while queue:
        current = queue.popleft()
        current_depth = distances[current]
        
        for neighbor in adj_list.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = current_depth + 1
                parent[neighbor] = current
                queue.append(neighbor)
                
                # Track level statistics
                level = current_depth + 1
                nodes_per_level[level] = nodes_per_level.get(level, 0) + 1
                max_depth = max(max_depth, level)
    
    # Compute average distance
    avg_distance = sum(distances.values()) / len(distances) if distances else 0
    
    return {
        "source": source,
        "visited_nodes": len(visited),
        "max_depth": max_depth,
        "average_distance": round(avg_distance, 2),
        "nodes_per_level": nodes_per_level,
        "reachable_ratio": len(visited) / num_nodes
    }


def lambda_handler(event, context):
    """
    Perform BFS on the input graph.
    
    Input: Graph data from GraphGenerator
    Output: BFS traversal results
    """

    # Streaming: Initialize publisher for incremental parameter streaming
    _streaming_session = (event.get('Session', '') if isinstance(event, dict) else '') or str(id(event))
    _streaming_publisher = StreamingPublisher(
        session_id=_streaming_session,
        source_function="BFSFunction",
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
    
    # Perform BFS from node 0
    result = compute_bfs(adj_list, num_nodes, source=0)
    _streaming_publisher.publish('result', result)
    
    end_time = datetime.datetime.now()
    compute_time = (end_time - start_time) / datetime.timedelta(microseconds=1)
    _streaming_publisher.publish('compute_time_us', compute_time)
    
    return {
        "algorithm": "BFS",
        "result": result,
        "compute_time_us": compute_time,
        "graph_nodes": num_nodes
    }
