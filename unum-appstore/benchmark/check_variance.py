import json, numpy as np

with open('results/run_20260308_204826/nlp-pipeline__Unum-Base.json') as f:
    base_data = json.load(f)
with open('results/run_20260308_204826/nlp-pipeline__Unum-Str.json') as f:
    str_data = json.load(f)

base_cold = [d['e2e_latency_ms'] for d in base_data if d['cold_start_count'] > 0]
base_warm = [d['e2e_latency_ms'] for d in base_data if d['cold_start_count'] == 0]
str_cold = [d['e2e_latency_ms'] for d in str_data if d['cold_start_count'] > 0]
str_warm = [d['e2e_latency_ms'] for d in str_data if d['cold_start_count'] == 0]

print(f"=== NLP Pipeline: Base vs Streaming ===")
print(f"Base cold: {len(base_cold)}, warm: {len(base_warm)}")
print(f"Str  cold: {len(str_cold)}, warm: {len(str_warm)}")
print()

bc = np.array(base_cold); sc = np.array(str_cold)
bw = np.array(base_warm); sw = np.array(str_warm)

print(f"Cold - Base: mean={np.mean(bc):.1f}, std={np.std(bc,ddof=1):.1f}, var={np.var(bc,ddof=1):.1f}")
print(f"Cold - Str:  mean={np.mean(sc):.1f}, std={np.std(sc,ddof=1):.1f}, var={np.var(sc,ddof=1):.1f}")
print(f"Cold variance reduction: {(1-np.var(sc,ddof=1)/np.var(bc,ddof=1))*100:.1f}%")
print()
print(f"Warm - Base: mean={np.mean(bw):.1f}, std={np.std(bw,ddof=1):.1f}, var={np.var(bw,ddof=1):.1f}")
print(f"Warm - Str:  mean={np.mean(sw):.1f}, std={np.std(sw,ddof=1):.1f}, var={np.var(sw,ddof=1):.1f}")
print(f"Warm variance reduction: {(1-np.var(sw,ddof=1)/np.var(bw,ddof=1))*100:.1f}%")
print()
print(f"Cold E2E improvement: {(np.mean(sc)-np.mean(bc))/np.mean(bc)*100:.1f}%")
print(f"Warm E2E improvement: {(np.mean(sw)-np.mean(bw))/np.mean(bw)*100:.1f}%")

# Function-level times
print("\n=== Warm avg function durations ===")
for label, data in [("Base", base_data), ("Str", str_data)]:
    cls, summ = [], []
    for e in data:
        if e['cold_start_count'] == 0:
            for fm in e['function_metrics']:
                if fm['function_name'] == 'Classifier': cls.append(fm['duration_ms'])
                elif fm['function_name'] == 'Summarizer': summ.append(fm['duration_ms'])
    print(f"  {label}: Classifier={np.mean(cls):.1f}ms, Summarizer={np.mean(summ):.1f}ms")
