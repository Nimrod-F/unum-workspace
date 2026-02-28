# Prompt for Generating New Fusion Test Workflow Scenarios

Use this prompt with any LLM (Claude, GPT, etc.) to generate new test workflows.
After generating, paste the output into `setup_scenarios.py` and re-run it.
Then run `python run_all_fusions.py` to test AI fusion on all workflows including new ones.

---

## The Prompt

```
I need you to generate new test workflow scenarios for my Unum serverless fusion testing framework.

Each workflow is a DAG of Lambda functions connected via unum_config.json files.
The fusion optimizer analyzes these workflows and recommends which sequential chains to merge.

### File structure per workflow

Each workflow lives in its own folder and needs:
1. `unum-template.yaml` - declares all functions with Properties (CodeUri, Runtime, MemorySize, Timeout)
2. For each function: a directory containing `unum_config.json` and `app.py`

### unum_config.json link types

- **Scalar (sequential 1-to-1):**
  {"Name": "X", "Next": {"Name": "Y", "InputType": "Scalar"}}

- **Fan-out (parallel invocation):**
  {"Name": "X", "Next": [{"Name": "Y", "InputType": "Scalar"}, {"Name": "Z", "InputType": "Scalar"}]}

- **Fan-in (aggregator waits for all branches):**
  {"Name": "Y", "Next": {"Name": "Agg", "InputType": {"Fan-in": {"Values": ["Y-unumIndex-0", "Z-unumIndex-1"]}}, "Fan-in-Group": true}}
  The index in `-unumIndex-N` matches the position in the parent's fan-out list.
  ALL terminal functions of each branch must have the SAME fan-in Next block pointing to the same aggregator.

- **Terminal (no successor):**
  {"Name": "X"} (no "Next" field)

### Fusion rules the optimizer follows

FUSE when: sequential 1-to-1 scalar chains (A->B->C all Scalar links)
DO NOT FUSE when:
1. Functions overlap between groups
2. Chain crosses a fan-in boundary (pre-fan-in function + aggregator)
3. Chain crosses a fan-out boundary (parent + parallel children)
4. Start function included in a chain
5. OR-node alternatives
6. Shared/reused functions across branches
7. Memory differs >2x between functions
8. Timeout already >300s
9. Different runtimes

### What I need from you

Generate a Python function (following the pattern below) for a NEW workflow scenario called `{SCENARIO_NAME}`.
The scenario should test: {DESCRIBE WHAT TOPOLOGY/EDGE CASE TO TEST}.

Use single-letter function names (A, B, C...) and keep descriptions brief.
Include varied MemorySize/Timeout values where relevant to test soft constraint rules.

### Pattern to follow (from setup_scenarios.py)

```python
def create_{scenario_name}():
    name = "{scenario-name}"
    # If there's a fan-in, define the values list:
    # fanin_vals = ["X-unumIndex-0", "Y-unumIndex-1"]

    functions = {
        "A": {"code_uri": "a/", "memory": 256, "timeout": 30, "start": True},
        "B": {"code_uri": "b/", "memory": 256, "timeout": 30},
        # ... more functions
    }

    configs = {
        "A": make_config("A", start=True, next_val=scalar_next("B")),
        # Fan-out: next_val=[scalar_next("B"), scalar_next("C")]
        # Scalar:  next_val=scalar_next("C")
        # Fan-in:  next_val=fanin_next("Agg", fanin_vals)
        # Terminal: (no next_val argument)
    }

    descriptions = {
        "A": "brief role description",
    }

    create_workflow(name, functions, configs, descriptions)
```

Helper functions available:
- `scalar_next(name)` - creates {"Name": name, "InputType": "Scalar"}
- `fanin_next(target, values)` - creates fan-in Next block
- `make_config(name, start=False, next_val=None)` - creates unum_config dict
- `create_workflow(name, functions, configs, descriptions)` - writes all files

### Important constraints
- Fan-in Values list must include ALL branch terminals, using format "{FuncName}-unumIndex-{N}"
  where N is the position of that branch in the parent's fan-out list
- ALL terminal functions of parallel branches must have identical fan-in Next blocks
- The `code_uri` must end with `/` and match a lowercase directory name (e.g., `"a/"`)
- Start function must have `"start": True` in functions dict AND `start=True` in make_config

Now generate the scenario.
```

---

## Example scenario requests to paste after the prompt

### Simple topologies
- "A diamond: A fans out to B and C, both fan-in to D. Tests that B and C stay separate."
- "A chain of 10 functions to test maximum fusion length."
- "Two completely independent chains triggered by the same Start function."

### Edge cases for fusion rules
- "A workflow where one branch has python3.11 and another has python3.13 (cross-runtime)."
- "A workflow with one function at Timeout: 600 to test near-timeout rule."
- "A workflow where function C is called by both branch A->B->C and branch D->E->C (shared function)."
- "A workflow with wildly different memory: 128MB, 256MB, 1024MB, 3008MB to test resource rule."

### Complex topologies
- "Nested fan-out: A fans to B,C; B fans to D,E; all fan-in to F."
- "Multi-stage: A->B->C fan-in to D, then D->E->F fan-out to G,H, fan-in to I."
- "A wide fan-out (6+ branches) with varying chain lengths per branch."
- "A workflow with back-to-back aggregators: branches merge into Agg1, then Agg1 fans out again to new branches that merge into Agg2."

---

## After generating

1. Paste the new function into `setup_scenarios.py`
2. Add a call to it in the `if __name__ == "__main__"` block
3. Add its diagram to the `DIAGRAMS` dict
4. Run: `python setup_scenarios.py`
5. Run: `python run_all_fusions.py --workflow {scenario-name}`
6. Or run all: `python run_all_fusions.py`
