# AI Fusion Analysis Results

Generated: 2026-02-21 21:31:08

---

## asymmetric-diamond

### Original Workflow

```
A (start) --> B --> C -----------\
          \-> D -----------------> G (aggregator)
          \-> E --> F ----------/
```

### LLM Response

```

============================================================
  Unum AI Fusion Advisor
============================================================

  Workflow:  asymmetric-diamond
  Functions: 7
  Template:  unum-template.yaml

  > A (start)
  > B
  > C
  > D
  > E
  > F
  > G

  Model:     gpt-4o

------------------------------------------------------------


ANALYSIS:
- Function A: Start function, fan-out to B, D, E.
- Function B: Sequentially calls C with InputType "Scalar".
- Function C: Terminal function in its path, feeds into G with Fan-in.
- Function D: Terminal function in its path, feeds into G with Fan-in.
- Function E: Sequentially calls F with InputType "Scalar".
- Function F: Terminal function in its path, feeds into G with Fan-in.
- Function G: Aggregator, receives inputs from C, D, F with Fan-in.


WORKFLOW GRAPH:
  A --> B --> C --\
      \-> D ------> G
      \-> E --> F --/


FUSION CANDIDATES:
- B -> C: Both functions have a Scalar "Next" link, same Runtime, and no fan-in/fan-out constraints. Qualifies for fusion under sequential 1-to-1 chain rule.


REJECTED CANDIDATES:
- A -> B: Rejected due to Rule 4 (Start function cannot be fused with downstream functions).
- D -> G: Rejected due to Rule 2 (cannot fuse across a fan-in boundary).
- E -> F: Rejected due to Rule 7 (MemorySize differs by more than 2x).


RECOMMENDATION:
fusions:
  - name: FusedBC
    chain:
      - B
      - C


EXPECTED IMPROVEMENTS:
- Reduced latency by eliminating the invocation overhead between B and C.
- Cost savings from reduced execution time and fewer invocations.
- Maintains parallelism and respects fan-in/fan-out boundaries.

------------------------------------------------------------

  Proposed fusion.yaml:

  fusions:
  - chain:
    - B
    - C
    name: FusedBC


  [Saved] fusion.yaml

[Fusion] Starting compilation from fusion.yaml...
  > Fusing: ['B', 'C'] -> FusedBC
  > Linking non-fused functions...
    [Patch] A: Next 'B' -> 'FusedBC'
    [Patch] D: Fan-in 'C-unumIndex-0' -> 'FusedBC-unumIndex-0'
    [Patch] F: Fan-in 'C-unumIndex-0' -> 'FusedBC-unumIndex-0'

[Success] Generated unum-template-fused.yaml
Functions patched and moved to fused_build/

  Fusion complete!


Scanning workflow   
                                                            

Analyzing workflow   
Analyzing workflow.  
Analyzing workflow.. 
Analyzing workflow...
Analyzing workflow   
Analyzing workflow.  
Analyzing workflow.. 
Analyzing workflow...
                                                            

Applying fusion   
                                                            

```

### Generated fusion.yaml

```yaml
fusions:
- chain:
  - B
  - C
  name: FusedBC

```

### Diagram

![asymmetric-diamond](diagrams\asymmetric-diamond.png)

---

## branching-pipeline

### Original Workflow

```
A (start) --> B --> C ----\
          \-> D --> E ----> F (aggregator)
```

### LLM Response

```

============================================================
  Unum AI Fusion Advisor
============================================================

  Workflow:  branching-pipeline
  Functions: 6
  Template:  unum-template.yaml

  > A (start)
  > B
  > C
  > D
  > E
  > F

  Model:     gpt-4o

------------------------------------------------------------


ANALYSIS:
- Function A: Start function, fan-out to B and D. It is the entry-point and splits into two paths.
- Function B: Part of the fast path, sequentially calls C. It is a simple 1-to-1 chain with C.
- Function C: Terminal of the fast path, feeds into F with a fan-in. It cannot be fused with F.
- Function D: Part of the slow path, sequentially calls E. It is a simple 1-to-1 chain with E.
- Function E: Terminal of the slow path, feeds into F with a fan-in. It cannot be fused with F.
- Function F: Aggregator, receives inputs from C and E via fan-in. It is the final step.


WORKFLOW GRAPH:
  A --> B --> C --\
       \-> D --> E --\ 
                      -> F


FUSION CANDIDATES:
- B -> C: Both functions are in a sequential 1-to-1 chain with Scalar input types. They share the same runtime and resource requirements, satisfying fusion rules for sequential chains.

- D -> E: Both functions are in a sequential 1-to-1 chain with Scalar input types. They share the same runtime and resource requirements, satisfying fusion rules for sequential chains.


REJECTED CANDIDATES:
- A -> B: Rejected because A is a Start function (Rule 4).
- C -> F: Rejected because C feeds into F via a fan-in (Rule 2).
- E -> F: Rejected because E feeds into F via a fan-in (Rule 2).
- A -> D: Rejected because A is a Start function (Rule 4).


RECOMMENDATION:
fusions:
  - name: FusedBC
    chain:
      - B
      - C
  - name: FusedDE
    chain:
      - D
      - E


EXPECTED IMPROVEMENTS:
- Reduced latency by eliminating invocation overhead between B and C, and between D and E.
- Cost savings from reduced number of Lambda invocations.
- Maintained parallelism and fan-in structure, ensuring no loss of functionality.

------------------------------------------------------------

  Proposed fusion.yaml:

  fusions:
  - chain:
    - B
    - C
    name: FusedBC
  - chain:
    - D
    - E
    name: FusedDE


  [Saved] fusion.yaml

[Fusion] Starting compilation from fusion.yaml...
  > Fusing: ['B', 'C'] -> FusedBC
  > Fusing: ['D', 'E'] -> FusedDE
  > Linking non-fused functions...
    [Patch] A: Next 'B' -> 'FusedBC'
    [Patch] A: Next 'D' -> 'FusedDE'

[Success] Generated unum-template-fused.yaml
Functions patched and moved to fused_build/

  Fusion complete!


Scanning workflow   
                                                            

Analyzing workflow   
Analyzing workflow.  
Analyzing workflow.. 
Analyzing workflow...
Analyzing workflow   
                                                            

Applying fusion   
                                                            

```

### Generated fusion.yaml

```yaml
fusions:
- chain:
  - B
  - C
  name: FusedBC
- chain:
  - D
  - E
  name: FusedDE

```

### Diagram

![branching-pipeline](diagrams\branching-pipeline.png)

---

## long-chain

### Original Workflow

```
A (start) -> B -> C -> D -> E -> F -> G
```

### LLM Response

```

============================================================
  Unum AI Fusion Advisor
============================================================

  Workflow:  long-chain
  Functions: 7
  Template:  unum-template.yaml

  > A (start)
  > B
  > C
  > D
  > E
  > F
  > G

  Model:     gpt-4o

------------------------------------------------------------


ANALYSIS:
- Function A: Start function, entry-point of the workflow, sequentially calls B.
- Function B: Sequentially called by A, calls C.
- Function C: Sequentially called by B, calls D.
- Function D: Sequentially called by C, calls E.
- Function E: Sequentially called by D, calls F.
- Function F: Sequentially called by E, calls G.
- Function G: Terminal function, sequentially called by F, stores final results.


WORKFLOW GRAPH:
  Start --> A --> B --> C --> D --> E --> F --> G


FUSION CANDIDATES:
- Candidate Chain: B -> C -> D -> E -> F -> G
  - All functions are linked by Scalar "Next" links.
  - All functions share the same Runtime (python3.13).
  - All functions have the same MemorySize (256 MB) and Timeout (30s).
  - This chain satisfies the fusion rule for sequential 1-to-1 chains.


REJECTED CANDIDATES:
- A -> B: Rejected due to Rule 4 (Start function A must remain standalone).


RECOMMENDATION:
fusions:
  - name: FusedBCDEFG
    chain:
      - B
      - C
      - D
      - E
      - F
      - G


EXPECTED IMPROVEMENTS:
- Reduced latency by eliminating inter-function invocation overhead between B, C, D, E, F, and G.
- Cost savings from reduced number of Lambda invocations.
- Simplified orchestration by reducing the number of functions to manage.

------------------------------------------------------------

  Proposed fusion.yaml:

  fusions:
  - chain:
    - B
    - C
    - D
    - E
    - F
    - G
    name: FusedBCDEFG


  [Saved] fusion.yaml

[Fusion] Starting compilation from fusion.yaml...
  > Fusing: ['B', 'C', 'D', 'E', 'F', 'G'] -> FusedBCDEFG
  > Linking non-fused functions...
    [Patch] A: Next 'B' -> 'FusedBCDEFG'

[Success] Generated unum-template-fused.yaml
Functions patched and moved to fused_build/

  Fusion complete!


Scanning workflow   
                                                            

Analyzing workflow   
Analyzing workflow.  
Analyzing workflow.. 
                                                            

Applying fusion   
                                                            

```

### Generated fusion.yaml

```yaml
fusions:
- chain:
  - B
  - C
  - D
  - E
  - F
  - G
  name: FusedBCDEFG

```

### Diagram

![long-chain](diagrams\long-chain.png)

---

## multi-aggregator

### Original Workflow

```
A (start) --> B ----------\
          \-> C -----------> D (aggregator) --> E --> F
```

### LLM Response

```

============================================================
  Unum AI Fusion Advisor
============================================================

  Workflow:  multi-aggregator
  Functions: 6
  Template:  unum-template.yaml

  > A (start)
  > B
  > C
  > D
  > E
  > F

  Model:     gpt-4o

------------------------------------------------------------


ANALYSIS:
- Function A: Start function, fan-out to B and C.
- Function B: Part of a fan-in with C, feeds into D.
- Function C: Part of a fan-in with B, feeds into D.
- Function D: Fan-in aggregator, sequentially calls E.
- Function E: Sequentially calls F.
- Function F: Terminal function, no further calls.


WORKFLOW GRAPH:
  A --> B --> D --> E --> F
    \-> C --/


FUSION CANDIDATES:
- Chain D -> E -> F: All links are Scalar, satisfying the sequential 1-to-1 chain rule. They share the same runtime and memory size, and do not cross any fan-in or fan-out boundaries.


REJECTED CANDIDATES:
- A -> B: Rejected due to Rule 3 (fan-out boundary). A is a fan-out function and cannot be fused with its parallel children.
- A -> C: Rejected due to Rule 3 (fan-out boundary). A is a fan-out function and cannot be fused with its parallel children.
- B -> D: Rejected due to Rule 2 (fan-in boundary). B is part of a fan-in and cannot be fused with the aggregator D.
- C -> D: Rejected due to Rule 2 (fan-in boundary). C is part of a fan-in and cannot be fused with the aggregator D.


RECOMMENDATION:
fusions:
  - name: FusedDEF
    chain:
      - D
      - E
      - F


EXPECTED IMPROVEMENTS:
- Reduced latency by eliminating invocation overhead between D, E, and F.
- Potential cost savings from fewer Lambda invocations.
- Maintains parallelism and respects fan-in/fan-out boundaries.

------------------------------------------------------------

  Proposed fusion.yaml:

  fusions:
  - chain:
    - D
    - E
    - F
    name: FusedDEF


  [Saved] fusion.yaml

[Fusion] Starting compilation from fusion.yaml...
  > Fusing: ['D', 'E', 'F'] -> FusedDEF
  > Linking non-fused functions...
    [Patch] B: Next 'D' -> 'FusedDEF'
    [Patch] C: Next 'D' -> 'FusedDEF'

[Success] Generated unum-template-fused.yaml
Functions patched and moved to fused_build/

  Fusion complete!


Scanning workflow   
                                                            

Analyzing workflow   
Analyzing workflow.  
Analyzing workflow.. 
Analyzing workflow...
                                                            

Applying fusion   
                                                            

```

### Generated fusion.yaml

```yaml
fusions:
- chain:
  - D
  - E
  - F
  name: FusedDEF

```

### Diagram

![multi-aggregator](diagrams\multi-aggregator.png)

---

## parallel-chains-merge

### Original Workflow

```
A (start) --> B --> C --> D ---------\
          \-> E --> F --> G ----------> H (aggregator) --> I
```

### LLM Response

```

============================================================
  Unum AI Fusion Advisor
============================================================

  Workflow:  parallel-chains-merge
  Functions: 9
  Template:  unum-template.yaml

  > A (start)
  > B
  > C
  > D
  > E
  > F
  > G
  > H
  > I

  Model:     gpt-4o

------------------------------------------------------------


ANALYSIS:
- Function A: Start function, fan-out to B and E.
- Function B: Sequentially calls C.
- Function C: Sequentially calls D.
- Function D: Terminal for its chain, feeds into fan-in at H.
- Function E: Sequentially calls F.
- Function F: Sequentially calls G.
- Function G: Terminal for its chain, feeds into fan-in at H.
- Function H: Fan-in aggregator, sequentially calls I.
- Function I: Terminal function, publishes final result.


WORKFLOW GRAPH:
  A --> B --> C --> D --\
       \                H --> I
        \-> E --> F --> G --/


FUSION CANDIDATES:
- B -> C: Both have Scalar "Next" links, same Runtime, MemorySize, and Timeout. Fusing reduces invocation overhead.
- E -> F: Both have Scalar "Next" links, same Runtime, MemorySize, and Timeout. Fusing reduces invocation overhead.


REJECTED CANDIDATES:
- A -> B: Rule 4 violation (A is a Start function).
- C -> D: Rule 2 violation (D feeds into a fan-in).
- F -> G: Rule 2 violation (G feeds into a fan-in).
- H -> I: Rule 7 violation (H has 512 MB, I has 256 MB).


RECOMMENDATION:
fusions:
  - name: FusedBC
    chain:
      - B
      - C
  - name: FusedEF
    chain:
      - E
      - F


EXPECTED IMPROVEMENTS:
- Reduced latency due to fewer function invocations in chains B->C and E->F.
- Lower cost from reduced invocation overhead.
- Maintained parallelism and fan-in/fan-out structure for optimal performance.

------------------------------------------------------------

  Proposed fusion.yaml:

  fusions:
  - chain:
    - B
    - C
    name: FusedBC
  - chain:
    - E
    - F
    name: FusedEF


  [Saved] fusion.yaml

[Fusion] Starting compilation from fusion.yaml...
  > Fusing: ['B', 'C'] -> FusedBC
  > Fusing: ['E', 'F'] -> FusedEF
  > Linking non-fused functions...
    [Patch] A: Next 'B' -> 'FusedBC'
    [Patch] A: Next 'E' -> 'FusedEF'

[Success] Generated unum-template-fused.yaml
Functions patched and moved to fused_build/

  Fusion complete!


Scanning workflow   
                                                            

Analyzing workflow   
Analyzing workflow.  
Analyzing workflow.. 
                                                            

Applying fusion   
                                                            

```

### Generated fusion.yaml

```yaml
fusions:
- chain:
  - B
  - C
  name: FusedBC
- chain:
  - E
  - F
  name: FusedEF

```

### Diagram

![parallel-chains-merge](diagrams\parallel-chains-merge.png)

---

