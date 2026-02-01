# Multi-Source Dashboard - File Manifest

## Complete File List

### Configuration Files (2)
- `unum-template.yaml` - UNUM application configuration with all 8 functions
- `unum-step-functions.json` - Step Functions workflow definition (fan-out/fan-in pattern)

### Lambda Functions (8 directories, 16 files)

#### Entry Point (1)
- `trigger-dashboard/app.py` - Entry function that triggers fan-out
- `trigger-dashboard/requirements.txt`

#### Data Fetch Functions (6)
- `fetch-sales-data/app.py` - Fast internal source (100-400ms)
- `fetch-sales-data/requirements.txt`
- `fetch-inventory-data/app.py` - Warehouse system (150-500ms)
- `fetch-inventory-data/requirements.txt`
- `fetch-marketing-metrics/app.py` - Marketing API (200-800ms)
- `fetch-marketing-metrics/requirements.txt`
- `fetch-external-market-data/app.py` - **SLOWEST** external API (500-3000ms)
- `fetch-external-market-data/requirements.txt`
- `fetch-weather-data/app.py` - Weather API (300-1500ms)
- `fetch-weather-data/requirements.txt`
- `fetch-competitor-pricing/app.py` - Web scraping (400-2500ms)
- `fetch-competitor-pricing/requirements.txt`

#### Aggregation Function (1)
- `merge-dashboard-data/app.py` - **KEY FILE** - Fan-in aggregation with sync+async support
- `merge-dashboard-data/requirements.txt`

### Benchmark Scripts (3 files)
- `benchmark/config.yaml` - Benchmark configuration (region, iterations, etc.)
- `benchmark/run_benchmark.py` - Main benchmark runner (auto-discovers functions)
- `benchmark/collect_metrics.py` - Metrics analysis and export (CSV, JSON)

### Documentation (3 files)
- `README.md` - Complete documentation (architecture, deployment, usage)
- `QUICKSTART.md` - Quick start guide (5-minute setup)
- `DEPLOYMENT_GUIDE.md` - Detailed deployment and benchmarking instructions

## File Statistics

- **Total files**: 24 (excluding this manifest)
- **Python files**: 8 Lambda functions + 2 benchmark scripts = 10
- **Config files**: 3 (YAML: 2, JSON: 1)
- **Documentation**: 3 (Markdown)
- **Requirements**: 8 (one per Lambda function)

## Key Files for Understanding

1. **Start here**: `README.md`
2. **Architecture**: `unum-step-functions.json` (see the parallel branches)
3. **Core logic**: `merge-dashboard-data/app.py` (fan-in implementation)
4. **Benchmarking**: `benchmark/run_benchmark.py`

## Generated Files (Not in Manifest)

After running `unum-cli compile`, the following files are auto-generated:

- `template.yaml` - AWS SAM template
- `*/unum_config.json` - Per-function UNUM configuration (8 files)
- `.aws-sam/` - Build artifacts directory

After running benchmarks:

- `benchmark/results/benchmark_*.json` - Benchmark results
- `benchmark/results/*.csv` - Exported analysis (if requested)
- `benchmark/charts/*.png` - Generated charts (if using chart generator)

## File Purposes

### Critical for Deployment
- unum-template.yaml
- unum-step-functions.json
- All app.py files

### Critical for Benchmarking
- benchmark/config.yaml
- benchmark/run_benchmark.py
- benchmark/collect_metrics.py

### Critical for Understanding
- README.md
- DEPLOYMENT_GUIDE.md
- merge-dashboard-data/app.py (demonstrates Future-Based pattern)

## Verification Checklist

Use this to verify all files exist:

```bash
# Check all Lambda functions
ls -1 */app.py | wc -l  # Should be 8

# Check all requirements.txt
ls -1 */requirements.txt | wc -l  # Should be 8

# Check config files
ls -1 *.yaml *.json  # Should show 2 files

# Check benchmark files
ls -1 benchmark/*.{py,yaml}  # Should show 3 files

# Check documentation
ls -1 *.md  # Should show 4 files (including this manifest)
```

## Lines of Code

Approximate breakdown:

- Lambda functions: ~800 lines total
- Benchmark scripts: ~600 lines total
- Documentation: ~1200 lines total
- **Total**: ~2600 lines

## Deployment Size

Approximate sizes:

- Per Lambda function: 5-10 KB (Python code only)
- Total deployment package: ~50-80 KB
- With dependencies: ~100-150 KB
- CloudFormation template: ~15 KB
