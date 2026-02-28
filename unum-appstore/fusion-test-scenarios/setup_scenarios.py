#!/usr/bin/env python3
"""Generate test workflow scenarios for AI fusion testing.

Creates 5 diverse workflow topologies with all files needed by unum-cli fuse --ai:
  - unum-template.yaml
  - {func_dir}/unum_config.json
  - {func_dir}/app.py (minimal stub)
"""
import os, json, yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def write_yaml(path, data):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def write_app_stub(path, func_name, description):
    with open(path, 'w') as f:
        f.write(f'def lambda_handler(event, context):\n')
        f.write(f'    """Function {func_name}: {description}"""\n')
        f.write(f'    import time\n')
        f.write(f'    time.sleep(0.1)\n')
        f.write(f'    result = event.copy() if isinstance(event, dict) else event\n')
        f.write(f'    return result\n')


def scalar_next(name):
    return {"Name": name, "InputType": "Scalar"}


def fanin_next(target, values):
    return {
        "Name": target,
        "InputType": {"Fan-in": {"Values": values}},
        "Fan-in-Group": True
    }


def make_config(name, start=False, next_val=None):
    cfg = {"Name": name, "Start": start, "Checkpoint": True, "Debug": True}
    if next_val is not None:
        cfg["Next"] = next_val
    return cfg


def make_template(app_name, functions):
    """Build unum-template.yaml dict.
    functions: dict of {Name: {code_uri, runtime, memory, timeout, start}}
    """
    template = {
        "Globals": {
            "ApplicationName": app_name,
            "Region": "eu-central-1",
            "FaaSPlatform": "aws",
            "UnumIntermediaryDataStoreType": "dynamodb",
            "UnumIntermediaryDataStoreName": f"unum-{app_name}",
            "Checkpoint": True,
            "GC": False,
            "Eager": True,
        },
        "Functions": {}
    }
    for name, props in functions.items():
        entry = {
            "Properties": {
                "CodeUri": props["code_uri"],
                "Runtime": props.get("runtime", "python3.13"),
                "MemorySize": props.get("memory", 256),
                "Timeout": props.get("timeout", 30),
            }
        }
        if props.get("start"):
            entry["Properties"]["Start"] = True
        template["Functions"][name] = entry
    return template


def create_workflow(name, functions, configs, descriptions):
    wf_dir = os.path.join(BASE_DIR, name)
    ensure_dir(wf_dir)

    template = make_template(name, functions)
    write_yaml(os.path.join(wf_dir, "unum-template.yaml"), template)

    for func_name, config in configs.items():
        code_uri = functions[func_name]["code_uri"]
        func_dir = os.path.join(wf_dir, code_uri.rstrip('/'))
        ensure_dir(func_dir)
        write_json(os.path.join(func_dir, "unum_config.json"), config)
        write_app_stub(
            os.path.join(func_dir, "app.py"),
            func_name,
            descriptions.get(func_name, "processing step")
        )

    # Write a README with the workflow diagram
    diagram = DIAGRAMS.get(name, "")
    with open(os.path.join(wf_dir, "README.md"), 'w') as f:
        f.write(f"# {name}\n\n")
        f.write(f"## Workflow Diagram\n\n```\n{diagram}\n```\n\n")
        f.write(f"## Functions ({len(functions)})\n\n")
        for fn in functions:
            desc = descriptions.get(fn, "")
            f.write(f"- **{fn}**: {desc}\n")
        f.write("\n")

    print(f"  Created: {name} ({len(functions)} functions)")


# ─────────────────────────────────────────────────────────────────────────────
# Workflow Diagrams (ASCII)
# ─────────────────────────────────────────────────────────────────────────────

DIAGRAMS = {
    "branching-pipeline": """\
A (start) --> B --> C ----\\
          \\-> D --> E ----> F (aggregator)""",

    "asymmetric-diamond": """\
A (start) --> B --> C -----------\\
          \\-> D -----------------> G (aggregator)
          \\-> E --> F ----------/""",

    "long-chain": """\
A (start) -> B -> C -> D -> E -> F -> G""",

    "multi-aggregator": """\
A (start) --> B ----------\\
          \\-> C -----------> D (aggregator) --> E --> F""",

    "parallel-chains-merge": """\
A (start) --> B --> C --> D ---------\\
          \\-> E --> F --> G ----------> H (aggregator) --> I""",
}


# ─────────────────────────────────────────────────────────────────────────────
# Workflow 1: branching-pipeline
#   A fans out to two branches (B->C and D->E), both fan-in to F
#   Expected fusion: B+C, D+E (sequential chains within parallel branches)
# ─────────────────────────────────────────────────────────────────────────────

def create_branching_pipeline():
    name = "branching-pipeline"
    fanin_vals = ["C-unumIndex-0", "E-unumIndex-1"]

    functions = {
        "A": {"code_uri": "a/", "memory": 256, "timeout": 30, "start": True},
        "B": {"code_uri": "b/", "memory": 256, "timeout": 30},
        "C": {"code_uri": "c/", "memory": 256, "timeout": 30},
        "D": {"code_uri": "d/", "memory": 256, "timeout": 30},
        "E": {"code_uri": "e/", "memory": 256, "timeout": 30},
        "F": {"code_uri": "f/", "memory": 512, "timeout": 60},
    }

    configs = {
        "A": make_config("A", start=True, next_val=[
            scalar_next("B"), scalar_next("D")
        ]),
        "B": make_config("B", next_val=scalar_next("C")),
        "C": make_config("C", next_val=fanin_next("F", fanin_vals)),
        "D": make_config("D", next_val=scalar_next("E")),
        "E": make_config("E", next_val=fanin_next("F", fanin_vals)),
        "F": make_config("F"),
    }

    descriptions = {
        "A": "receive request and split into fast/slow processing paths",
        "B": "fast path: initial processing",
        "C": "fast path: finalize and prepare for aggregation",
        "D": "slow path: initial processing with heavier computation",
        "E": "slow path: finalize and prepare for aggregation",
        "F": "aggregate results from both processing paths",
    }

    create_workflow(name, functions, configs, descriptions)


# ─────────────────────────────────────────────────────────────────────────────
# Workflow 2: asymmetric-diamond
#   A fans out to 3 branches: B->C (chain), D (direct), E->F (chain)
#   All fan-in to G
#   Expected fusion: B+C, E+F (chains), D stays alone (single node)
# ─────────────────────────────────────────────────────────────────────────────

def create_asymmetric_diamond():
    name = "asymmetric-diamond"
    fanin_vals = ["C-unumIndex-0", "D-unumIndex-1", "F-unumIndex-2"]

    functions = {
        "A": {"code_uri": "a/", "memory": 256, "timeout": 30, "start": True},
        "B": {"code_uri": "b/", "memory": 256, "timeout": 30},
        "C": {"code_uri": "c/", "memory": 256, "timeout": 30},
        "D": {"code_uri": "d/", "memory": 128, "timeout": 10},
        "E": {"code_uri": "e/", "memory": 512, "timeout": 60},
        "F": {"code_uri": "f/", "memory": 512, "timeout": 30},
        "G": {"code_uri": "g/", "memory": 512, "timeout": 60},
    }

    configs = {
        "A": make_config("A", start=True, next_val=[
            scalar_next("B"), scalar_next("D"), scalar_next("E")
        ]),
        "B": make_config("B", next_val=scalar_next("C")),
        "C": make_config("C", next_val=fanin_next("G", fanin_vals)),
        "D": make_config("D", next_val=fanin_next("G", fanin_vals)),
        "E": make_config("E", next_val=scalar_next("F")),
        "F": make_config("F", next_val=fanin_next("G", fanin_vals)),
        "G": make_config("G"),
    }

    descriptions = {
        "A": "receive data and distribute to enrichment services",
        "B": "query primary database",
        "C": "format primary database results",
        "D": "quick cache lookup (lightweight, low memory)",
        "E": "call external API (heavy, high memory)",
        "F": "parse and normalize external API response",
        "G": "merge all enriched data sources",
    }

    create_workflow(name, functions, configs, descriptions)


# ─────────────────────────────────────────────────────────────────────────────
# Workflow 3: long-chain
#   Pure sequential chain: A -> B -> C -> D -> E -> F -> G
#   Expected fusion: entire chain (or large segments) into one function
# ─────────────────────────────────────────────────────────────────────────────

def create_long_chain():
    name = "long-chain"

    functions = {
        "A": {"code_uri": "a/", "memory": 256, "timeout": 30, "start": True},
        "B": {"code_uri": "b/", "memory": 256, "timeout": 30},
        "C": {"code_uri": "c/", "memory": 256, "timeout": 30},
        "D": {"code_uri": "d/", "memory": 256, "timeout": 30},
        "E": {"code_uri": "e/", "memory": 256, "timeout": 30},
        "F": {"code_uri": "f/", "memory": 256, "timeout": 30},
        "G": {"code_uri": "g/", "memory": 256, "timeout": 30},
    }

    configs = {
        "A": make_config("A", start=True, next_val=scalar_next("B")),
        "B": make_config("B", next_val=scalar_next("C")),
        "C": make_config("C", next_val=scalar_next("D")),
        "D": make_config("D", next_val=scalar_next("E")),
        "E": make_config("E", next_val=scalar_next("F")),
        "F": make_config("F", next_val=scalar_next("G")),
        "G": make_config("G"),
    }

    descriptions = {
        "A": "ingest raw data from source",
        "B": "validate data schema",
        "C": "normalize field values",
        "D": "apply business rule transformations",
        "E": "compute derived aggregates",
        "F": "format output for target system",
        "G": "store final results",
    }

    create_workflow(name, functions, configs, descriptions)


# ─────────────────────────────────────────────────────────────────────────────
# Workflow 4: multi-aggregator
#   A fans out to B and C, both fan-in to D, then D -> E -> F (chain)
#   Expected fusion: E+F (post-aggregation chain); B and C stay separate
# ─────────────────────────────────────────────────────────────────────────────

def create_multi_aggregator():
    name = "multi-aggregator"
    fanin_vals = ["B-unumIndex-0", "C-unumIndex-1"]

    functions = {
        "A": {"code_uri": "a/", "memory": 256, "timeout": 30, "start": True},
        "B": {"code_uri": "b/", "memory": 256, "timeout": 30},
        "C": {"code_uri": "c/", "memory": 256, "timeout": 30},
        "D": {"code_uri": "d/", "memory": 256, "timeout": 30},
        "E": {"code_uri": "e/", "memory": 256, "timeout": 30},
        "F": {"code_uri": "f/", "memory": 256, "timeout": 30},
    }

    configs = {
        "A": make_config("A", start=True, next_val=[
            scalar_next("B"), scalar_next("C")
        ]),
        "B": make_config("B", next_val=fanin_next("D", fanin_vals)),
        "C": make_config("C", next_val=fanin_next("D", fanin_vals)),
        "D": make_config("D", next_val=scalar_next("E")),
        "E": make_config("E", next_val=scalar_next("F")),
        "F": make_config("F"),
    }

    descriptions = {
        "A": "receive search query, split into sub-queries",
        "B": "search database alpha",
        "C": "search database beta",
        "D": "merge and deduplicate search results",
        "E": "rank merged results by relevance",
        "F": "format and return final ranked results",
    }

    create_workflow(name, functions, configs, descriptions)


# ─────────────────────────────────────────────────────────────────────────────
# Workflow 5: parallel-chains-merge
#   A fans out to two 3-step chains (B->C->D and E->F->G), both fan-in to H,
#   then H -> I
#   Expected fusion: B+C+D, E+F+G (parallel chain segments), H+I
# ─────────────────────────────────────────────────────────────────────────────

def create_parallel_chains_merge():
    name = "parallel-chains-merge"
    fanin_vals = ["D-unumIndex-0", "G-unumIndex-1"]

    functions = {
        "A": {"code_uri": "a/", "memory": 256, "timeout": 30, "start": True},
        "B": {"code_uri": "b/", "memory": 256, "timeout": 30},
        "C": {"code_uri": "c/", "memory": 256, "timeout": 30},
        "D": {"code_uri": "d/", "memory": 256, "timeout": 30},
        "E": {"code_uri": "e/", "memory": 256, "timeout": 30},
        "F": {"code_uri": "f/", "memory": 256, "timeout": 30},
        "G": {"code_uri": "g/", "memory": 256, "timeout": 30},
        "H": {"code_uri": "h/", "memory": 512, "timeout": 60},
        "I": {"code_uri": "i/", "memory": 256, "timeout": 30},
    }

    configs = {
        "A": make_config("A", start=True, next_val=[
            scalar_next("B"), scalar_next("E")
        ]),
        "B": make_config("B", next_val=scalar_next("C")),
        "C": make_config("C", next_val=scalar_next("D")),
        "D": make_config("D", next_val=fanin_next("H", fanin_vals)),
        "E": make_config("E", next_val=scalar_next("F")),
        "F": make_config("F", next_val=scalar_next("G")),
        "G": make_config("G", next_val=fanin_next("H", fanin_vals)),
        "H": make_config("H", next_val=scalar_next("I")),
        "I": make_config("I"),
    }

    descriptions = {
        "A": "receive image, split into processing and metadata paths",
        "B": "resize image to target dimensions",
        "C": "apply color correction and white balance",
        "D": "apply artistic filters and finalize image",
        "E": "extract EXIF and metadata from original",
        "F": "analyze image content with ML model",
        "G": "generate descriptive tags from analysis",
        "H": "combine processed image with enriched metadata",
        "I": "publish final result to storage",
    }

    create_workflow(name, functions, configs, descriptions)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating fusion test scenarios...\n")
    create_branching_pipeline()
    create_asymmetric_diamond()
    create_long_chain()
    create_multi_aggregator()
    create_parallel_chains_merge()
    print(f"\nDone! Created 5 workflows in {BASE_DIR}")
