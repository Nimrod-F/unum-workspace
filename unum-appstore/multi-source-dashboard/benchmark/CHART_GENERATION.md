# Chart Generation Guide

## Overview

The `generate_charts.py` script creates visualization charts specifically for the multi-source-dashboard benchmark results, comparing CLASSIC, EAGER, and FUTURE_BASED execution modes.

## Usage

### Generate from specific results file

```bash
cd benchmark
python generate_charts.py --results-file results/benchmark_20260201_205450.json
```

### Generate from latest file in results directory

```bash
python generate_charts.py --results-dir results/
```

### Custom output directory

```bash
python generate_charts.py --results-file results/benchmark_20260201_205450.json --output-dir my_charts/
```

## Generated Charts

The script generates 6 different visualizations:

### 1. **e2e_latency_comparison.png**
Bar chart comparing end-to-end latency across modes with error bars.
- Shows mean latency ± standard deviation
- Color-coded: Red (CLASSIC), Orange (EAGER), Green (FUTURE_BASED)
- Best for: Quick visual comparison of overall performance

### 2. **cold_vs_warm_comparison.png**
Grouped bar chart showing cold start vs warm start impact.
- Red bars: Cold start latencies
- Blue bars: Warm start latencies
- Best for: Understanding cold start penalty across modes

### 3. **improvement_over_classic.png**
Bar chart showing percentage improvement vs CLASSIC baseline.
- Green bars: Improvements (faster than CLASSIC)
- Red bars: Regressions (slower than CLASSIC)
- Shows exact percentage values
- Best for: Quantifying the benefit of FUTURE_BASED mode

### 4. **pre_resolved_efficiency.png**
Bar chart showing background polling efficiency in FUTURE_BASED mode.
- Green: Pre-resolved inputs (already cached by background polling)
- Red: Blocked inputs (had to wait for completion)
- Out of 6 total parallel inputs
- Best for: Demonstrating future-based execution efficiency

### 5. **latency_distribution.png**
Box and whisker plot showing latency distribution.
- Shows median (dark red line), mean (blue dashed line)
- Box: 25th to 75th percentile
- Whiskers: Min to max range
- Best for: Understanding variance and outliers

### 6. **summary_table.png**
Comprehensive table with all key metrics.
- E2E latency (mean, std dev, min, max)
- Cold vs warm start latencies
- Success rate
- Pre-resolved count
- Improvement percentages
- Best for: Complete numerical summary

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
boto3
pyyaml
matplotlib
numpy
```

## Example Workflow

```bash
# 1. Run benchmark
cd benchmark
python run_benchmark.py --mode all --iterations 10

# 2. Generate charts from latest results
python generate_charts.py --results-dir results/

# 3. View charts
cd charts
ls -la *.png
```

## Output Directory Structure

```
benchmark/
├── results/
│   └── benchmark_20260201_205450.json
├── charts/                              # Generated charts
│   ├── e2e_latency_comparison.png
│   ├── cold_vs_warm_comparison.png
│   ├── improvement_over_classic.png
│   ├── pre_resolved_efficiency.png
│   ├── latency_distribution.png
│   └── summary_table.png
├── generate_charts.py
├── run_benchmark.py
└── collect_metrics.py
```

## Interpreting Results

### Good Results
✅ FUTURE_BASED E2E latency < CLASSIC E2E latency
✅ Improvement over CLASSIC: 15-25%
✅ Pre-resolved inputs: 4-5 out of 6 (67-83%)
✅ Low variance (std dev < 15% of mean)

### Issues to Investigate
⚠️ No improvement or negative improvement
⚠️ Pre-resolved count = 0 (futures not working)
⚠️ Very high variance (std dev > 25% of mean)
⚠️ Cold start latency much higher than expected

## Customization

To customize the charts, edit `generate_charts.py`:

- **Colors**: Modify `color_map` dictionaries
- **Chart sizes**: Adjust `figsize` parameters
- **Fonts**: Change `fontsize` parameters
- **DPI**: Modify `dpi=300` in `savefig()` calls

## Troubleshooting

### ModuleNotFoundError: No module named 'matplotlib'

```bash
pip install matplotlib numpy
```

### UnicodeEncodeError on Windows

The script has been fixed to avoid Unicode characters. If you still see this, ensure:
```bash
chcp 65001  # Set console to UTF-8
```

### Charts look wrong

Check that your results file has data for all modes:
```bash
python -c "import json; print(json.load(open('results/benchmark_XXX.json')).keys())"
# Should show: dict_keys(['CLASSIC', 'EAGER', 'FUTURE_BASED'])
```

### No improvement shown

If FUTURE_BASED shows no improvement:
1. Check that functions have `UNUM_FUTURE_BASED=true` environment variable
2. Verify `EAGER=true` is also set
3. Check DynamoDB table has checkpoints
4. Review CloudWatch logs for errors

## Integration with Research Paper

The charts are designed to be publication-ready:
- **High DPI (300)**: Suitable for papers
- **Clear labels**: No manual annotation needed
- **Professional styling**: Follows academic standards
- **Multiple views**: Different perspectives on same data

Recommended charts for paper:
1. Primary: `e2e_latency_comparison.png` or `latency_distribution.png`
2. Secondary: `improvement_over_classic.png`
3. Supporting: `pre_resolved_efficiency.png`
4. Appendix: `summary_table.png`

## Comparison with Generic Chart Generator

| Feature | Multi-Source Generator | Generic Benchmark Generator |
|---------|----------------------|----------------------------|
| Purpose | Single app, multiple modes | Multiple apps comparison |
| Input | Single JSON file | Directory of results |
| Charts | 6 focused charts | 5 comparison charts |
| Pre-resolved | ✅ Dedicated chart | ❌ Not shown |
| Customization | App-specific | Generic |

Use **this generator** for multi-source-dashboard results.
Use the **generic generator** (`../benchmark/generate_comparison_charts.py`) when comparing multiple workflows.
