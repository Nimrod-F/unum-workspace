"""
Aggregator Function - Collects and summarizes results from all graph algorithms

This is the fan-in point that receives results from PageRank, BFS, and MST.
"""
import datetime


def lambda_handler(event, context):
    """
    Aggregate results from all parallel graph analysis functions.
    
    Input: Array of results from [PageRank, BFS, MST]
    Output: Comprehensive summary report
    """
    start_time = datetime.datetime.now()
    
    # Handle different input formats
    results = event if isinstance(event, list) else [event]
    
    # Initialize summary
    summary = {
        "status": "completed",
        "algorithms_processed": 0,
        "total_compute_time_us": 0,
        "results": {}
    }
    
    # Process each result
    for result in results:
        if not isinstance(result, dict):
            continue
            
        algorithm = result.get("algorithm", "unknown")
        compute_time = result.get("compute_time_us", 0)
        algo_result = result.get("result", {})
        graph_nodes = result.get("graph_nodes", 0)
        
        summary["results"][algorithm] = {
            "compute_time_us": compute_time,
            "details": algo_result
        }
        summary["algorithms_processed"] += 1
        summary["total_compute_time_us"] += compute_time
        
        if graph_nodes > 0:
            summary["graph_nodes"] = graph_nodes
    
    # Add analysis insights
    if summary["algorithms_processed"] > 0:
        summary["insights"] = generate_insights(summary["results"])
    
    end_time = datetime.datetime.now()
    aggregation_time = (end_time - start_time) / datetime.timedelta(microseconds=1)
    summary["aggregation_time_us"] = aggregation_time
    
    return summary


def generate_insights(results):
    """Generate insights from the combined analysis results."""
    insights = []
    
    # PageRank insights
    if "PageRank" in results:
        pr_details = results["PageRank"]["details"]
        if pr_details.get("converged"):
            insights.append(f"PageRank converged in {pr_details.get('iterations', 'N/A')} iterations")
        top_nodes = pr_details.get("top_nodes", [])
        if top_nodes:
            insights.append(f"Most influential nodes: {top_nodes[:3]}")
    
    # BFS insights
    if "BFS" in results:
        bfs_details = results["BFS"]["details"]
        max_depth = bfs_details.get("max_depth", 0)
        reachable = bfs_details.get("reachable_ratio", 0)
        insights.append(f"Graph diameter (from source): {max_depth}")
        if reachable < 1.0:
            insights.append(f"Warning: Only {reachable*100:.1f}% of nodes reachable from source")
        else:
            insights.append("Graph is fully connected from source node")
    
    # MST insights
    if "MST" in results:
        mst_details = results["MST"]["details"]
        total_weight = mst_details.get("total_weight", 0)
        is_connected = mst_details.get("is_connected", False)
        insights.append(f"MST total weight: {total_weight}")
        if not is_connected:
            insights.append("Warning: Graph is not fully connected")
    
    return insights
