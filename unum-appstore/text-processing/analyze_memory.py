import json

# Load results
with open('benchmark_results/classic_research_results.json') as f:
    classic = json.load(f)

with open('benchmark_results/future_research_results.json') as f:
    future = json.load(f)

print("CLASSIC - Functions captured per iteration:")
print("-" * 60)
for r in classic['results']:
    funcs = list(r.get('lambda_metrics', {}).keys())
    mem = r.get('total_memory_used_mb', 0)
    print(f"  Iter {r['iteration']}: {len(funcs)} functions, Memory={mem}MB")
    print(f"    Functions: {funcs}")

print("\nFUTURE_BASED - Functions captured per iteration:")
print("-" * 60)
for r in future['results']:
    funcs = list(r.get('lambda_metrics', {}).keys())
    mem = r.get('total_memory_used_mb', 0)
    print(f"  Iter {r['iteration']}: {len(funcs)} functions, Memory={mem}MB")
    print(f"    Functions: {funcs}")

print("\n" + "=" * 60)
print("ANALYSIS:")
print("=" * 60)

classic_func_counts = [len(r.get('lambda_metrics', {})) for r in classic['results']]
future_func_counts = [len(r.get('lambda_metrics', {})) for r in future['results']]

print(f"CLASSIC avg functions captured: {sum(classic_func_counts)/len(classic_func_counts):.1f}")
print(f"FUTURE avg functions captured: {sum(future_func_counts)/len(future_func_counts):.1f}")

classic_mems = [r.get('total_memory_used_mb', 0) for r in classic['results']]
future_mems = [r.get('total_memory_used_mb', 0) for r in future['results']]

print(f"\nCLASSIC memory values: {classic_mems}")
print(f"FUTURE memory values: {future_mems}")
