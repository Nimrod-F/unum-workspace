"""
Stage 4: Reporter Function (Final Stage)

Produces final reports for each analyzed item and a summary.
This is the last stage, so no streaming output needed.
"""

import json
import time

ITEM_TIME = 0.5  # Each report takes 0.5 seconds to generate


def generate_report(item_id, item_data):
    """Generate report for a single item - simulates 0.5s of computation."""
    start = time.time()
    
    # Extract values from input item
    if isinstance(item_data, dict):
        analyzed_value = item_data.get("analyzed_value", 0)
        score = item_data.get("score", 0)
        category = item_data.get("category", "unknown")
    else:
        analyzed_value = 0
        score = 0
        category = "unknown"
    
    result = {
        "id": item_id,
        "source": "reporter",
        "final_value": analyzed_value,
        "final_score": score,
        "category": category,
        "grade": "A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40 else "D",
        "reported_at": time.time()
    }
    
    # Ensure minimum processing time
    elapsed = time.time() - start
    if elapsed < ITEM_TIME:
        time.sleep(ITEM_TIME - elapsed)
    
    return result


def lambda_handler(event, context):
    """Reporter handler - generates final reports."""
    
    start_time = time.time()
    print(f"[Reporter] Starting")
    
    reports = {}
    
    # Generate report for analyzed_1
    t1 = time.time()
    analyzed_1 = event.get("analyzed_1")
    wait_1 = time.time() - t1
    print(f"[Reporter] Got analyzed_1 in {wait_1:.3f}s")
    reports["report_1"] = generate_report(1, analyzed_1)
    print(f"[Reporter] report_1 ready at {time.time() - start_time:.3f}s")
    
    # Generate report for analyzed_2
    t2 = time.time()
    analyzed_2 = event.get("analyzed_2")
    wait_2 = time.time() - t2
    print(f"[Reporter] Got analyzed_2 in {wait_2:.3f}s")
    reports["report_2"] = generate_report(2, analyzed_2)
    print(f"[Reporter] report_2 ready at {time.time() - start_time:.3f}s")
    
    # Generate report for analyzed_3
    t3 = time.time()
    analyzed_3 = event.get("analyzed_3")
    wait_3 = time.time() - t3
    print(f"[Reporter] Got analyzed_3 in {wait_3:.3f}s")
    reports["report_3"] = generate_report(3, analyzed_3)
    print(f"[Reporter] report_3 ready at {time.time() - start_time:.3f}s")
    
    # Generate report for analyzed_4
    t4 = time.time()
    analyzed_4 = event.get("analyzed_4")
    wait_4 = time.time() - t4
    print(f"[Reporter] Got analyzed_4 in {wait_4:.3f}s")
    reports["report_4"] = generate_report(4, analyzed_4)
    print(f"[Reporter] report_4 ready at {time.time() - start_time:.3f}s")
    
    # Generate report for analyzed_5
    t5 = time.time()
    analyzed_5 = event.get("analyzed_5")
    wait_5 = time.time() - t5
    print(f"[Reporter] Got analyzed_5 in {wait_5:.3f}s")
    reports["report_5"] = generate_report(5, analyzed_5)
    print(f"[Reporter] report_5 ready at {time.time() - start_time:.3f}s")
    
    # Generate summary
    total_time = time.time() - start_time
    
    summary = {
        "total_items": 5,
        "total_score": sum(r.get("final_score", 0) for r in reports.values()),
        "average_score": sum(r.get("final_score", 0) for r in reports.values()) / 5,
        "grades": {r["grade"]: sum(1 for x in reports.values() if x.get("grade") == r["grade"]) 
                   for r in reports.values()},
        "processing_time": total_time,
        "completed_at": time.time()
    }
    
    print(f"[Reporter] Completed in {total_time:.3f}s")
    print(f"[Reporter] Summary: {json.dumps(summary)}")
    
    return {
        "reports": reports,
        "summary": summary
    }
