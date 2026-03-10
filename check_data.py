import json, glob

# Text-processing earlier research results
with open('unum-appstore/text-processing/benchmark_results/classic_research_results.json') as f:
    classic = json.load(f)
with open('unum-appstore/text-processing/benchmark_results/future_research_results.json') as f:
    future = json.load(f)

print('=== TEXT-PROCESSING (earlier research benchmark) ===')
for mode_name, data in [('CLASSIC', classic), ('FUTURE', future)]:
    cold_runs = [r for r in data['runs'] if r.get('cold_starts', 0) > 0]
    warm_runs = [r for r in data['runs'] if r.get('cold_starts', 0) == 0]
    print(f'{mode_name}: {len(data["runs"])} total, {len(cold_runs)} cold, {len(warm_runs)} warm')
    if warm_runs:
        warm_e2e = [r['e2e_latency_ms'] for r in warm_runs]
        warm_cost = [r.get('estimated_cost_usd', 0) for r in warm_runs]
        print(f'  Warm E2E: mean={sum(warm_e2e)/len(warm_e2e):.0f}, all={warm_e2e}')
    if cold_runs:
        cold_e2e = [r['e2e_latency_ms'] for r in cold_runs]
        print(f'  Cold E2E: mean={sum(cold_e2e)/len(cold_e2e):.0f}, all={cold_e2e}')

# WordCount all earlier results
print('\n=== WORDCOUNT EARLIER RUNS ===')
for ts in ['084230', '085700']:
    try:
        with open(f'unum-appstore/wordcount/benchmark/results/benchmark_CLASSIC_20260116_{ts}_runs.json') as f:
            d = json.load(f)
        e2es = [r['e2e_latency_ms'] for r in d]
        print(f'WordCount CLASSIC {ts}: {len(d)} runs, E2Es={e2es}, mean={sum(e2es)/len(e2es):.0f}')
    except Exception as e:
        print(f'Error {ts}: {e}')

for ts in ['090726', '092058', '093738', '094914', '100240']:
    try:
        with open(f'unum-appstore/wordcount/benchmark/results/benchmark_FUTURE_BASED_20260116_{ts}_runs.json') as f:
            d = json.load(f)
        e2es = [r['e2e_latency_ms'] for r in d]
        print(f'WordCount FUTURE {ts}: {len(d)} runs, E2Es={e2es}, mean={sum(e2es)/len(e2es):.0f}')
    except Exception as e:
        print(f'Error {ts}: {e}')

# Order processing - with cold/warm separation
print('\n=== ORDER-PROCESSING 3-MODES ===')
for mode in ['CLASSIC', 'FUTURE_BASED']:
    try:
        with open(f'unum-appstore/order-processing-workflow/benchmark/results_3-modes/results/benchmark_order-processing-workflow_{mode}_20260205_211306_runs.json' if mode == 'CLASSIC' else f'unum-appstore/order-processing-workflow/benchmark/results_3-modes/results/benchmark_order-processing-workflow_{mode}_20260205_211935_runs.json') as f:
            d = json.load(f)
        cold_runs = [r for r in d if r.get('cold_starts', 0) > 0]
        warm_runs = [r for r in d if r.get('cold_starts', 0) == 0]
        print(f'{mode}: total={len(d)}, cold={len(cold_runs)}, warm={len(warm_runs)}')
        if warm_runs:
            e2es = [r['e2e_latency_ms'] for r in warm_runs]
            print(f'  Warm E2E: {e2es}, mean={sum(e2es)/len(e2es):.0f}')
        if cold_runs:
            e2es = [r['e2e_latency_ms'] for r in cold_runs]
            print(f'  Cold E2E: {e2es}, mean={sum(e2es)/len(e2es):.0f}')
    except Exception as e:
        print(f'Error {mode}: {e}')

# Order processing - Fusion results
print('\n=== ORDER-PROCESSING FUSION ===')
try:
    with open('unum-appstore/order-processing-workflow/benchmark/results-fusion/benchmark_runs_20260206_003949.json') as f:
        d = json.load(f)
    for mode_name, runs in d.items():
        cold_runs = [r for r in runs if r.get('cold_starts', 0) > 0]
        warm_runs = [r for r in runs if r.get('cold_starts', 0) == 0]
        print(f'{mode_name}: total={len(runs)}, cold={len(cold_runs)}, warm={len(warm_runs)}')
        if warm_runs:
            e2es = [r['e2e_latency_ms'] for r in warm_runs]
            print(f'  Warm E2E: {e2es}, mean={sum(e2es)/len(e2es):.0f}')
        if cold_runs:
            e2es = [r['e2e_latency_ms'] for r in cold_runs]
            print(f'  Cold E2E: {e2es}, mean={sum(e2es)/len(e2es):.0f}')
except Exception as e:
    print(f'Error: {e}')

# Graph analysis earlier
print('\n=== GRAPH-ANALYSIS EARLIER ===')
for fname in glob.glob('unum-appstore/graph-analysis/benchmark/results/benchmark_*.json'):
    with open(fname) as f:
        d = json.load(f)
    print(f'\nFile: {fname}')
    for mode in ['classic', 'future_based']:
        if mode in d:
            runs = d[mode]['runs']
            e2es = [r['e2e_latency_ms'] for r in runs]
            cold_starts = [r.get('cold_starts', 0) for r in runs]
            print(f'  {mode}: {len(runs)} runs, E2Es={[round(e,0) for e in e2es]}, colds={cold_starts}')

# Monte Carlo formal results - check Str config
print('\n=== MONTE CARLO FORMAL RESULTS ===')
import os
base = 'unum-appstore/benchmark/results/run_20260308_204826'
for config in ['Base', 'Fus', 'Str', 'Fut', 'All']:
    fn = os.path.join(base, f'montecarlo-pipeline__Unum-{config}.json')
    if os.path.exists(fn):
        with open(fn) as f:
            data = json.load(f)
        cold = [d['e2e_latency_ms'] for d in data[:3]]
        warm = [d['e2e_latency_ms'] for d in data[3:]]
        warm_cost = [d['estimated_cost_usd'] for d in data[3:]]
        warm_billed = [d['total_billed_duration_ms'] for d in data[3:]]
        print(f'MC {config}: cold_avg={sum(cold)/len(cold):.0f} warm_avg={sum(warm)/len(warm):.0f} warm_billed={sum(warm_billed)/len(warm_billed):.0f} warm_cost=${sum(warm_cost)/len(warm_cost)*1e5:.2f}e-5')
