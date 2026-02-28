#!/usr/bin/env python3
"""
AI Fusion Benchmark Runner
===========================
Discovers all workflow directories, runs unum-cli fuse --ai for each,
captures LLM responses, and generates comparison diagrams.

Usage:
    python run_all_fusions.py                  # run everything
    python run_all_fusions.py --diagrams-only  # skip AI calls, just regenerate diagrams
    python run_all_fusions.py --workflow long-chain  # run only one workflow
"""
import os, sys, json, yaml, re, subprocess, argparse, textwrap
from datetime import datetime
from pathlib import Path

# --- Diagram imports ---------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Warning] matplotlib not installed. Diagrams will be skipped.")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("[Warning] networkx not installed. Diagrams will be skipped.")

# --- Paths -------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
CLI_PATH = SCRIPT_DIR.parent.parent / "unum" / "unum-cli" / "unum-cli.py"
RESULTS_DIR = SCRIPT_DIR / "results"
DIAGRAMS_DIR = RESULTS_DIR / "diagrams"


def strip_ansi(text):
    """Remove ANSI escape codes from text."""
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


# --- Workflow Discovery ------------------------------------------------------

def discover_workflows(base_dir=None, filter_name=None):
    """Find all subdirectories containing unum-template.yaml."""
    base = Path(base_dir) if base_dir else SCRIPT_DIR
    workflows = []
    for item in sorted(base.iterdir()):
        if item.is_dir() and (item / "unum-template.yaml").exists():
            if filter_name and item.name != filter_name:
                continue
            workflows.append(item)
    return workflows


# --- Graph Building ----------------------------------------------------------

def build_graph_from_configs(workflow_dir):
    """Parse unum_config.json files and build a directed graph dict.
    Returns: {nodes: [...], edges: [(src, dst, label), ...], start: str}
    """
    template_path = workflow_dir / "unum-template.yaml"
    with open(template_path, 'r') as f:
        template = yaml.safe_load(f)

    functions = template.get("Functions", {})
    nodes = []
    edges = []
    start_node = None
    node_props = {}

    for func_name, func_def in functions.items():
        props = func_def.get("Properties", {})
        code_uri = props.get("CodeUri", "")
        config_path = workflow_dir / code_uri.rstrip('/') / "unum_config.json"

        nodes.append(func_name)
        node_props[func_name] = {
            "memory": props.get("MemorySize", 256),
            "timeout": props.get("Timeout", 30),
            "runtime": props.get("Runtime", "python3.13"),
        }

        if props.get("Start"):
            start_node = func_name

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

            if config.get("Start"):
                start_node = func_name

            next_val = config.get("Next")
            if next_val is None:
                continue

            if isinstance(next_val, dict):
                target = next_val["Name"]
                itype = next_val.get("InputType", "Scalar")
                label = "fan-in" if isinstance(itype, dict) else ""
                edges.append((func_name, target, label))
            elif isinstance(next_val, list):
                for entry in next_val:
                    target = entry["Name"]
                    edges.append((func_name, target, ""))

    # Deduplicate edges
    seen = set()
    unique_edges = []
    for e in edges:
        key = (e[0], e[1])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    return {
        "nodes": nodes,
        "edges": unique_edges,
        "start": start_node,
        "props": node_props,
    }


def build_fused_graph(workflow_dir):
    """Build graph from the fused template if it exists."""
    fused_template_path = workflow_dir / "unum-template-fused.yaml"
    if not fused_template_path.exists():
        return None

    with open(fused_template_path, 'r') as f:
        template = yaml.safe_load(f)

    functions = template.get("Functions", {})
    nodes = list(functions.keys())
    edges = []
    start_node = None

    # Read configs from fused_build or original locations
    for func_name, func_def in functions.items():
        props = func_def.get("Properties", {})
        code_uri = props.get("CodeUri", "")

        if props.get("Start"):
            start_node = func_name

        config_path = workflow_dir / code_uri.rstrip('/') / "unum_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

            if config.get("Start"):
                start_node = func_name

            next_val = config.get("Next")
            if next_val is None:
                continue
            if isinstance(next_val, dict):
                target = next_val["Name"]
                itype = next_val.get("InputType", "Scalar")
                label = "fan-in" if isinstance(itype, dict) else ""
                edges.append((func_name, target, label))
            elif isinstance(next_val, list):
                for entry in next_val:
                    edges.append((func_name, entry["Name"], ""))

    seen = set()
    unique_edges = []
    for e in edges:
        key = (e[0], e[1])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    return {
        "nodes": nodes,
        "edges": unique_edges,
        "start": start_node,
    }


def parse_fusion_yaml(workflow_dir):
    """Read fusion.yaml and return the fusion groups."""
    fusion_path = workflow_dir / "fusion.yaml"
    if not fusion_path.exists():
        return []
    with open(fusion_path, 'r') as f:
        data = yaml.safe_load(f)
    if data and "fusions" in data:
        return data["fusions"]
    return []


# --- Diagram Drawing ---------------------------------------------------------

def hierarchical_layout(graph_data):
    """Compute a left-to-right hierarchical layout using topological ordering."""
    G = nx.DiGraph()
    G.add_nodes_from(graph_data["nodes"])
    for src, dst, _ in graph_data["edges"]:
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst)

    # Assign layers via longest path from start
    start = graph_data.get("start") or (graph_data["nodes"][0] if graph_data["nodes"] else None)
    layers = {}

    if start and start in G.nodes:
        # BFS to assign layers
        from collections import deque
        queue = deque([(start, 0)])
        visited = {start}
        layers[start] = 0

        while queue:
            node, layer = queue.popleft()
            for succ in G.successors(node):
                new_layer = layer + 1
                if succ not in layers or new_layer > layers[succ]:
                    layers[succ] = new_layer
                if succ not in visited:
                    visited.add(succ)
                    queue.append((succ, new_layer))

    # Handle disconnected nodes
    for n in G.nodes:
        if n not in layers:
            layers[n] = 0

    # Group by layer
    layer_groups = {}
    for node, layer in layers.items():
        layer_groups.setdefault(layer, []).append(node)

    # Assign positions
    max_layer = max(layers.values()) if layers else 0
    pos = {}
    for layer, nodes_in_layer in layer_groups.items():
        n = len(nodes_in_layer)
        for i, node in enumerate(sorted(nodes_in_layer)):
            x = layer * 2.0
            y = -(i - (n - 1) / 2.0) * 1.5
            pos[node] = (x, y)

    return pos


def draw_graph(ax, graph_data, title, fused_groups=None, is_fused_result=False):
    """Draw a workflow graph on a matplotlib axes."""
    if not graph_data or not graph_data["nodes"]:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11, fontweight='bold')
        return

    G = nx.DiGraph()
    G.add_nodes_from(graph_data["nodes"])
    for src, dst, label in graph_data["edges"]:
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst, label=label)

    pos = hierarchical_layout(graph_data)

    # Color nodes
    COLORS = ['#4FC3F7', '#81C784', '#FFB74D', '#E57373', '#BA68C8',
              '#4DD0E1', '#AED581', '#FF8A65', '#F06292', '#7986CB']

    node_colors = []
    if fused_groups:
        # Build a map: function -> group index
        func_to_group = {}
        for idx, group in enumerate(fused_groups):
            chain = group.get("chain", [])
            for fn in chain:
                func_to_group[fn] = idx

        for node in G.nodes:
            if node in func_to_group:
                node_colors.append(COLORS[func_to_group[node] % len(COLORS)])
            else:
                node_colors.append('#E0E0E0')
    elif is_fused_result:
        for node in G.nodes:
            if node.startswith("Fused"):
                node_colors.append('#81C784')
            else:
                node_colors.append('#E0E0E0')
    else:
        start = graph_data.get("start")
        for node in G.nodes:
            if node == start:
                node_colors.append('#4FC3F7')
            elif G.out_degree(node) == 0:
                node_colors.append('#FFB74D')
            else:
                node_colors.append('#E0E0E0')

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1800, node_color=node_colors,
                           edgecolors='#333333', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')

    # Draw edges
    edge_colors = []
    for u, v in G.edges():
        label = G.edges[u, v].get('label', '')
        edge_colors.append('#E57373' if label == 'fan-in' else '#666666')

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           arrows=True, arrowsize=20, arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.1', width=1.5)

    # Draw fan-in labels
    edge_labels = {(u, v): G.edges[u, v].get('label', '')
                   for u, v in G.edges() if G.edges[u, v].get('label')}
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=7,
                                     font_color='#C62828')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.axis('off')


def generate_diagram(workflow_dir, workflow_name):
    """Generate a 3-panel comparison diagram for a workflow."""
    if not HAS_MATPLOTLIB or not HAS_NETWORKX:
        return None

    original_graph = build_graph_from_configs(workflow_dir)
    fused_groups = parse_fusion_yaml(workflow_dir)
    fused_graph = build_fused_graph(workflow_dir)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Fusion Analysis: {workflow_name}", fontsize=14, fontweight='bold', y=0.98)

    # Panel 1: Original workflow
    draw_graph(axes[0], original_graph, "Original Workflow")

    # Panel 2: LLM suggestion (original graph with fusion groups highlighted)
    if fused_groups:
        draw_graph(axes[1], original_graph, "LLM Fusion Suggestion", fused_groups=fused_groups)
        # Add legend
        COLORS = ['#4FC3F7', '#81C784', '#FFB74D', '#E57373', '#BA68C8',
                  '#4DD0E1', '#AED581', '#FF8A65', '#F06292', '#7986CB']
        legend_patches = []
        for idx, group in enumerate(fused_groups):
            name = group.get("name", f"Group {idx}")
            chain = " -> ".join(group.get("chain", []))
            legend_patches.append(
                mpatches.Patch(color=COLORS[idx % len(COLORS)],
                               label=f"{name}: {chain}")
            )
        legend_patches.append(mpatches.Patch(color='#E0E0E0', label='Unchanged'))
        axes[1].legend(handles=legend_patches, loc='lower center',
                       fontsize=7, framealpha=0.9, ncol=1)
    else:
        draw_graph(axes[1], original_graph, "LLM: No Fusions Suggested")

    # Panel 3: Fused result
    if fused_graph:
        draw_graph(axes[2], fused_graph, "After Fusion", is_fused_result=True)
    else:
        axes[2].text(0.5, 0.5, "No fused result\n(fusion not applied)",
                     ha='center', va='center', transform=axes[2].transAxes,
                     fontsize=11, color='gray')
        axes[2].set_title("After Fusion", fontsize=11, fontweight='bold')
        axes[2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIAGRAMS_DIR / f"{workflow_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return out_path


# --- AI Fusion Runner --------------------------------------------------------

def run_ai_fusion(workflow_dir):
    """Run unum-cli fuse --ai -y on a workflow and capture output."""
    cmd = [
        sys.executable, str(CLI_PATH),
        "fuse", "--ai", "-y",
        "-t", "unum-template.yaml"
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(workflow_dir),
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ}  # inherit env (for OPENAI_API_KEY / .env)
        )
        output = result.stdout + result.stderr
        return strip_ansi(output), result.returncode
    except subprocess.TimeoutExpired:
        return "[ERROR] Command timed out after 120s", 1
    except Exception as e:
        return f"[ERROR] {e}", 1


# --- Markdown Report ---------------------------------------------------------

def generate_report(results):
    """Generate a markdown report with all LLM responses."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "fusion_responses.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# AI Fusion Analysis Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"---\n\n")

        for wf_name, data in results.items():
            f.write(f"## {wf_name}\n\n")

            # Original diagram (ASCII)
            readme_path = SCRIPT_DIR / wf_name / "README.md"
            if readme_path.exists():
                with open(readme_path, 'r') as r:
                    readme = r.read()
                    # Extract the diagram block
                    diagram_match = re.search(r'```\n(.*?)\n```', readme, re.DOTALL)
                    if diagram_match:
                        f.write(f"### Original Workflow\n\n```\n{diagram_match.group(1)}\n```\n\n")

            f.write(f"### LLM Response\n\n")
            f.write(f"```\n{data['response']}\n```\n\n")

            # Fusion result
            fusion_path = SCRIPT_DIR / wf_name / "fusion.yaml"
            if fusion_path.exists():
                with open(fusion_path, 'r') as fy:
                    f.write(f"### Generated fusion.yaml\n\n```yaml\n{fy.read()}\n```\n\n")
            else:
                f.write(f"### Generated fusion.yaml\n\nNo fusions recommended.\n\n")

            if data.get("diagram_path"):
                rel_path = os.path.relpath(data["diagram_path"], RESULTS_DIR)
                f.write(f"### Diagram\n\n![{wf_name}]({rel_path})\n\n")

            f.write(f"---\n\n")

    print(f"\n  Report saved to: {report_path}")
    return report_path


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run AI fusion analysis on all test workflows")
    parser.add_argument("--diagrams-only", action="store_true",
                        help="Skip AI calls, only regenerate diagrams from existing results")
    parser.add_argument("--workflow", type=str, default=None,
                        help="Run only a specific workflow by name")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Base directory to scan for workflows (default: script directory)")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  AI Fusion Benchmark Runner")
    print("=" * 60)
    print()

    workflows = discover_workflows(args.base_dir, args.workflow)

    if not workflows:
        print("  No workflows found!")
        sys.exit(1)

    print(f"  Found {len(workflows)} workflow(s):")
    for wf in workflows:
        print(f"    > {wf.name}")
    print()

    results = {}

    for wf_dir in workflows:
        wf_name = wf_dir.name
        print(f"{'-' * 60}")
        print(f"  Processing: {wf_name}")
        print(f"{'-' * 60}")

        if not args.diagrams_only:
            print(f"  Running AI fusion...")
            # Clean previous fusion artifacts
            for cleanup in ["fusion.yaml", "unum-template-fused.yaml"]:
                p = wf_dir / cleanup
                if p.exists():
                    p.unlink()
            fused_build = wf_dir / "fused_build"
            if fused_build.exists():
                import shutil
                shutil.rmtree(fused_build)

            response, returncode = run_ai_fusion(wf_dir)
            print(f"  Exit code: {returncode}")

            if returncode != 0:
                print(f"  [Warning] AI fusion returned non-zero exit code")
        else:
            # Load existing response if available
            response = "(diagrams-only mode, no LLM response captured)"

        # Generate diagram
        diagram_path = None
        if HAS_MATPLOTLIB and HAS_NETWORKX:
            print(f"  Generating diagram...")
            diagram_path = generate_diagram(wf_dir, wf_name)
            if diagram_path:
                print(f"  Diagram: {diagram_path}")

        results[wf_name] = {
            "response": response,
            "diagram_path": diagram_path,
        }

        print()

    # Generate markdown report
    print(f"{'-' * 60}")
    print(f"  Generating report...")
    report_path = generate_report(results)

    print()
    print("=" * 60)
    print(f"  Done! Processed {len(workflows)} workflow(s)")
    print(f"  Report:   {report_path}")
    print(f"  Diagrams: {DIAGRAMS_DIR}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
