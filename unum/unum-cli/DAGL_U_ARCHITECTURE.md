# DAGL-U: Architecture, Language Specification & Runtime Contract

> This document describes the implemented DAGL-U language, its compilation pipeline,
> the intermediate representation it targets, and the runtime execution model.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Language Definition (DAGL-U)](#2-language-definition-dagl-u)
3. [Compilation Pipeline](#3-compilation-pipeline)
4. [Abstract Syntax Tree (AST) Contract](#4-abstract-syntax-tree-ast-contract)
5. [Intermediate Representation (Unum IR)](#5-intermediate-representation-unum-ir)
6. [Runtime Execution Model](#6-runtime-execution-model)
7. [Denotational Semantics Sketch](#7-denotational-semantics-sketch)
8. [File Reference](#8-file-reference)
9. [Examples](#9-examples)

---

## 1. System Overview

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        COMPILE TIME                                  │
│                                                                      │
│  workflow.dag ──→ Tokenizer ──→ Token[] ──→ Parser ──→ AST (JSON)   │
│                   (lexical)                 (recursive               │
│                                              descent +               │
│                                              precedence              │
│                                              climbing)               │
│                          AST ──→ IR Generator ──→ unum_config.json  │
│                                                   (per function)     │
│                                  ──→ unum-template.yaml              │
├──────────────────────────────────────────────────────────────────────┤
│                        DEPLOY TIME                                   │
│                                                                      │
│  unum-template.yaml ──→ SAM/GCP/Azure Template ──→ Cloud Deploy     │
│  unum_config.json    ──→ Bundled into each function package          │
├──────────────────────────────────────────────────────────────────────┤
│                        RUNTIME (Decentralized)                       │
│                                                                      │
│  Each function:                                                      │
│    1. Reads its own unum_config.json                                 │
│    2. Executes user code                                             │
│    3. Checkpoints result to datastore (exactly-once)                 │
│    4. Invokes next function(s) per its config                        │
│    5. No centralized orchestrator — only FaaS + Datastore            │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Property

The language eliminates the centralized orchestrator. Unlike AWS Step Functions
(which requires a state machine service), DAGL-U compiles to per-function
configs that enable **decentralized orchestration** — each function knows its
own successor(s) and handles invocation, checkpointing, and fan-in coordination
autonomously.

---

## 2. Language Definition (DAGL-U)

### 2.1 Lexical Elements

| Category     | Tokens                                                                    |
| ------------ | ------------------------------------------------------------------------- |
| Keywords     | `let`, `return`, `if`, `else`, `map`, `in`, `collect`, `into`, `parallel` |
| Booleans     | `true`, `false`                                                           |
| Logical      | `not`, `and`, `or`                                                        |
| Literals     | `NUMBER` (42, 3.14, 2.5e10), `STRING` ("...", '...')                      |
| Identifiers  | `[a-zA-Z_][a-zA-Z0-9_]*`                                                  |
| Action Paths | `/path/to/action` (legacy DAGL compatibility)                             |
| Directives   | `@name` (e.g., `@workflow`, `@checkpoint`, `@eager`, `@aws`, `@gcp`)      |
| Operators    | `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `=`            |
| Lambda       | `\` (backslash), `->` (arrow)                                             |
| Delimiters   | `(`, `)`, `{`, `}`, `[`, `]`, `,`, `:`, `;`, `.`                          |
| Comments     | `// line comment`, `/* block comment */`                                  |

### 2.2 Grammar (PEG-style)

```peg
Program        ← Directive* Body
Directive      ← '@' Identifier '(' Expression ')'
Body           ← Block / MultiExpression

Block          ← Statement*
Statement      ← LetAssign / BareAssign / Return / Collect / Expression
LetAssign      ← 'let' Identifier '=' Expression
BareAssign     ← Identifier '=' Expression
Return         ← 'return' Expression
Collect        ← 'collect' Expression 'into' Expression

MultiExpression← Expression+

Expression     ← PlatformAnnot? LogicalOr
PlatformAnnot  ← '@aws' / '@gcp' / '@azure'

LogicalOr      ← LogicalAnd ('or' LogicalAnd)*
LogicalAnd     ← Equality ('and' Equality)*
Equality       ← Comparison (('==' / '!=') Comparison)*
Comparison     ← Addition (('<' / '<=' / '>' / '>=') Addition)*
Addition       ← Multiplication (('+' / '-') Multiplication)*
Multiplication ← Unary (('*' / '/' / '%') Unary)*
Unary          ← ('not' / '-') Unary / Postfix

Postfix        ← Primary (Index / Call / DotAccess)*
Index          ← '[' Expression ']'
Call           ← '(' Arguments ')'
DotAccess      ← '.' Identifier

Primary        ← Number / String / Boolean / Identifier / ActionPath
               / ArrayLiteral / ObjectLiteral / ParenExpr
               / IfExpr / MapExpr / ParallelExpr / LambdaExpr

IfExpr         ← 'if' Expression '{' Block '}' ('else' ('{' Block '}' / IfExpr))?
MapExpr        ← 'map' Identifier 'in' Expression ('{' Block '}' / Expression)
ParallelExpr   ← 'parallel' '{' Expression* '}'
LambdaExpr     ← '\' Identifier '->' Expression

Arguments      ← NamedArgs / PositionalArgs / ε
NamedArgs      ← Identifier ':' Expression (',' Identifier ':' Expression)*
PositionalArgs ← Expression (',' Expression)*

ArrayLiteral   ← '[' (Expression (',' Expression)*)? ']'
ObjectLiteral  ← '{' (Identifier ':' Expression (',' Identifier ':' Expression)*)? '}'

Number         ← [0-9]+ ('.' [0-9]+)? ([eE] [+-]? [0-9]+)?
String         ← '"' ... '"' / "'" ... "'"
Boolean        ← 'true' / 'false'
Identifier     ← [a-zA-Z_] [a-zA-Z0-9_]*
ActionPath     ← '/' [a-zA-Z0-9_/-]+
```

### 2.3 Operator Precedence (highest → lowest)

| Level | Operators                | Associativity |
| ----- | ------------------------ | ------------- |
| 1     | Primary, `()`, `[]`, `.` | Left          |
| 2     | `not`, unary `-`         | Right         |
| 3     | `*`, `/`, `%`            | Left          |
| 4     | `+`, `-`                 | Left          |
| 5     | `<`, `<=`, `>`, `>=`     | Left          |
| 6     | `==`, `!=`               | Left          |
| 7     | `and`                    | Left          |
| 8     | `or`                     | Left          |

### 2.4 DAGL-U Extensions over Original DAGL

| Feature              | DAGL (original)    | DAGL-U (extended)                                        |
| -------------------- | ------------------ | -------------------------------------------------------- |
| Function references  | `/_/action()`      | `FunctionName()` (platform-agnostic)                     |
| Workflow metadata    | None               | `@workflow("name")`, `@checkpoint(true)`, `@eager(true)` |
| Fan-in               | Not supported      | `collect <source> into <target>()`                       |
| Explicit parallel    | Auto-detected only | `parallel { expr₁ expr₂ ... }` + auto-detection          |
| Platform annotations | Not supported      | `@aws`, `@gcp`, `@azure` per-expression                  |
| Type declarations    | Not supported      | (Planned) `@function F(x: T) -> U`                       |

---

## 3. Compilation Pipeline

### 3.1 Pipeline Stages

```
Source Code (.dag)
    │
    ▼ ──── tokenize(source) ────────────────── dagl_compiler.py
Token Stream (Token[])
    │
    ▼ ──── Parser(tokens).parse() ──────────── dagl_compiler.py
Abstract Syntax Tree (dict)
    │
    ▼ ──── WorkflowGraph().compile(ast) ─────── dagl_to_unum_ir.py
Unum IR (dict[str, dict])
    │
    ├──→ unum_config.json (one per function)
    └──→ unum-template.yaml (workflow metadata)
```

### 3.2 Tokenizer Contract

**Input**: `source: str` — raw DAGL-U source code

**Output**: `list[Token]` where `Token = {type: str, value: str, pos: int}`

**Invariants**:

- Whitespace and comments are discarded (not in output)
- Token order preserved from source
- `pos` is the byte offset in the original source
- All tokens have one of the types listed in §2.1
- Tokenization is greedy (longest match first)
- On failure: raises `SyntaxError` with position

### 3.3 Parser Contract

**Input**: `list[Token]`

**Output**: AST `dict` (see §4 for complete node type spec)

**Invariants**:

- Parser is a **recursive-descent** parser with **operator-precedence climbing**
- No left recursion
- Deterministic (no backtracking)
- On ambiguity: directives before body; `if-else` binds to nearest `if`
- On failure: raises `SyntaxError` with token position and context

**Key parsing decisions**:

- If the program starts with `let`, `return`, `collect`, or `IDENTIFIER =` → parse as Block
- Otherwise → parse as multi-expression (potential parallel auto-detection)
- Parallel auto-detection: if ALL top-level expressions are invocations and count > 1, wrap in `list` node (parallel execution)

### 3.4 IR Generator Contract

**Input**: AST `dict` (top-level `block_expr`)

**Output**: `dict[str, dict]` — mapping function names to `unum_config.json` content

**Invariants**:

- Each unique function invocation in the AST becomes one entry in the output
- Data flow is tracked via variable bindings:
  - `let x = F()` → `x` maps to function `F`
  - `G(x)` where `x` maps to `F` → creates edge `F → G`
- Map expressions generate synthetic entry functions (`UnumMap0`, `UnumMap1`, ...)
- Fan-in edges use wildcard patterns (`FuncName-unumIndex-*`)
- Exactly one function has `Start: true` (the first in topological order)
- Terminal functions have no `Next` field
- Platform annotations propagate to `Platform` field in config

---

## 4. Abstract Syntax Tree (AST) Contract

### 4.1 Node Structure

Every AST node is a JSON object:

```json
{
  "data": "<node_type>",
  "children": [<child₁>, <child₂>, ...]
}
```

Children can be:

- Other AST nodes (dicts with `data` and `children`)
- Primitive values: `str`, `float`, `int`

### 4.2 Node Types — Complete Specification

#### Structural Nodes

| Node Type    | Children                     | Semantics                              |
| ------------ | ---------------------------- | -------------------------------------- |
| `block_expr` | `[stmt₁, stmt₂, ..., stmtₙ]` | Sequential execution of statements     |
| `assign`     | `[id_node, expr]`            | Bind `expr` result to variable `id`    |
| `return`     | `[expr]`                     | Return `expr` as block/workflow output |
| `directive`  | `[name: str, value: expr]`   | Workflow metadata (`@name(value)`)     |

#### Invocation Nodes

| Node Type    | Children                        | Semantics                                  |
| ------------ | ------------------------------- | ------------------------------------------ |
| `invocation` | `[name: str, args: dict\|list]` | Call function `name` with arguments `args` |
| `apply`      | `[fn_expr, arg_expr]`           | Apply lambda/variable to argument          |

#### Control Flow Nodes

| Node Type  | Children                         | Semantics                                     |
| ---------- | -------------------------------- | --------------------------------------------- |
| `if_expr`  | `[cond, then_block, else_block]` | If `cond` then `then_block` else `else_block` |
| `map_expr` | `[var: str, iterable, body]`     | Map `body` over each element of `iterable`    |
| `collect`  | `[source_expr, target_invoc]`    | Fan-in: gather `source` results into `target` |
| `parallel` | `[expr₁, expr₂, ..., exprₙ]`     | Execute all expressions in parallel           |

#### Expression Nodes

| Node Type  | Children                 | Semantics                                     |
| ---------- | ------------------------ | --------------------------------------------- |
| `binop`    | `[left, op: str, right]` | Binary operation (`+`, `-`, `==`, `and`, ...) |
| `unop`     | `[op: str, operand]`     | Unary operation (`not`, `-`)                  |
| `index`    | `[expr, idx_expr]`       | Subscript access (`expr[idx]`)                |
| `lambda`   | `[param: str, body]`     | Lambda function (`\param -> body`)            |
| `platform` | `[provider: str, expr]`  | Platform annotation (`@aws expr`)             |

#### Literal Nodes

| Node Type | Children                | Semantics                             |
| --------- | ----------------------- | ------------------------------------- |
| `number`  | `[value: float]`        | Numeric literal                       |
| `string`  | `[value: str]`          | String literal                        |
| `id`      | `[name: str]`           | Identifier or action path reference   |
| `list`    | `[elem₁, elem₂, ...]`   | Array literal OR parallel invocations |
| `dict`    | `[pair₁, pair₂, ...]`   | Object literal OR named arguments     |
| `pair`    | `[id_node, value_expr]` | Key-value pair in `dict`              |

### 4.3 Dual Role of `list` Node

The `list` node serves two purposes:

1. **Array literal**: `[1, 2, 3]` → `{data: "list", children: [num(1), num(2), num(3)]}`
2. **Parallel execution marker**: When all children are `invocation` nodes at the top level, it signals parallel execution to the IR generator

This duality is intentional: in the functional semantics, parallel execution IS
collection construction — the result is a list of all branch outputs.

---

## 5. Intermediate Representation (Unum IR)

### 5.1 Per-Function Config Schema

```typescript
interface UnumConfig {
  Name: string; // Function identifier
  Start: boolean; // Is this the workflow entry point?
  Checkpoint: boolean; // Enable exactly-once checkpointing?
  Debug: boolean; // Enable debug logging?

  // Optional: describes the outgoing edge(s)
  Next?: NextConfig | NextConfig[];

  // Optional: payload transformations before invoking Next
  "Next Payload Modifiers"?: string[]; // e.g., ["Pop"]

  // Optional: cross-platform annotation (DAGL-U extension)
  Platform?: "aws" | "gcp" | "azure";
}

interface NextConfig {
  Name: string; // Target function name
  InputType: InputType; // How data flows to the target
  Conditional?: string; // Runtime condition (e.g., "$out > 50")
  "Fan-in-Group"?: boolean; // Is this a fan-in coordination group?
}

type InputType =
  | "Scalar" // 1:1 — pass output directly
  | "Map" // 1:N — fan-out over array elements
  | { "Fan-in": { Values: string[] } }; // N:1 — collect from branches
```

### 5.2 InputType Semantics

| InputType  | Direction | Runtime Behavior                                                                |
| ---------- | --------- | ------------------------------------------------------------------------------- |
| `"Scalar"` | 1→1       | Output of source becomes input of target                                        |
| `"Map"`    | 1→N       | Source output is an array; one target instance per element                      |
| `"Fan-in"` | N→1       | Multiple sources collect into single target; last-to-arrive triggers invocation |

### 5.3 Fan-in Coordination Protocol

When `InputType` is `Fan-in`:

1. Each source function writes its output to the datastore with a unique key (e.g., `Mapper-unumIndex-0`)
2. Each source atomically increments a counter in the datastore
3. The last source to increment (counter == total) invokes the fan-in target
4. The target receives all source outputs as an ordered array
5. `"Next Payload Modifiers": ["Pop"]` strips the fan-out metadata before invoking the target

Wildcard `*` in Values (e.g., `"Mapper-unumIndex-*"`) expands at runtime based on the `Fan-out.Size` field in the payload.

### 5.4 AST → IR Mapping Rules

These are the compilation rules the IR generator implements:

| DAGL-U Pattern                                | Unum IR Pattern                                                          |
| --------------------------------------------- | ------------------------------------------------------------------------ |
| `let x = F()` then `G(x)`                     | `F.Next = {Name: "G", InputType: "Scalar"}`                              |
| `let x = F()` then `let y = G(x)` then `H(y)` | Chain: `F→G→H` all Scalar                                                |
| `map item in input { F(item) }`               | Synthetic `UnumMapN.Next = {Name: "F", InputType: "Map"}`                |
| `let x = map ... { F() }` then `G(x)`         | `F.Next = {Name: "G", InputType: {Fan-in: {Values: ["F-unumIndex-*"]}}}` |
| `parallel { A() B() C() }` from `prev`        | `prev.Next = [{A, Scalar}, {B, Scalar}, {C, Scalar}]`                    |
| `collect branches into Agg()`                 | Each branch: `.Next = {Name: "Agg", InputType: {Fan-in: ...}}`           |
| `@aws F()`                                    | `F.Platform = "aws"` (extension field)                                   |
| `@workflow("name")`                           | `template.Globals.ApplicationName = "name"`                              |
| `@checkpoint(true/false)`                     | `template.Globals.Checkpoint = true/false`                               |
| `@eager(true)`                                | `template.Globals.Eager = true`                                          |

---

## 6. Runtime Execution Model

### 6.1 Per-Function Execution Contract

Every deployed function wraps user code with the Unum runtime (`unum.py`):

```python
# Pseudocode of runtime contract
def lambda_handler(event, context):
    config = load("unum_config.json")
    unum = Unum(config, datastore, platform)

    # 1. CHECKPOINT CHECK (Exactly-once)
    existing = unum.get_checkpoint(event.Session, instance_name)
    if existing:
        return existing  # Skip — already executed

    # 2. INGRESS — Extract user data from Unum payload
    user_input = unum.ingress(event)

    # 3. EXECUTE — Run user function
    user_output = user_function(user_input, context)

    # 4. CHECKPOINT — Atomic write (first writer wins)
    unum.checkpoint_if_not_exist(event.Session, instance_name, user_output)

    # 5. EGRESS — Determine and invoke next function(s)
    unum.egress(user_output, event)
    return user_output
```

### 6.2 Payload Structure

Every function receives this payload:

```json
{
  "Data": {
    "Source": "http | dynamodb",
    "Value": "<user_data> | [<checkpoint_references>]"
  },
  "Session": "<uuid4>",
  "Fan-out": {
    "Index": 0,
    "Size": 5,
    "OuterLoop": { "Index": 1, "Size": 3 }
  }
}
```

| Field               | Type                     | Purpose                                                    |
| ------------------- | ------------------------ | ---------------------------------------------------------- |
| `Data.Source`       | `"http"` \| `"dynamodb"` | Where the input data comes from                            |
| `Data.Value`        | any                      | The actual input data or datastore references              |
| `Session`           | `str (uuid4)`            | Unique workflow execution ID (propagated to all functions) |
| `Fan-out.Index`     | `int`                    | This instance's index in a fan-out (0-based)               |
| `Fan-out.Size`      | `int`                    | Total number of parallel instances                         |
| `Fan-out.OuterLoop` | `Fan-out?`               | Nested fan-out context (for nested maps)                   |

### 6.3 Invocation Naming Convention

Each function invocation has a globally unique name:

```
<FunctionName>-UnumIndex-<level₀>.<level₁>.<level₂>...
```

- `level₀` = innermost fan-out index
- `level₁` = next outer fan-out index
- Dots separate nesting levels

**Example**: `Mapper-UnumIndex-2.1` = Mapper, 2nd element of inner fan-out, 1st element of outer fan-out.

### 6.4 Datastore Interface

```python
class DataStore(ABC):
    def checkpoint_data(self, session, key, data) -> bool:
        """Atomically write data if key doesn't exist. Returns True if written."""

    def get_checkpoint(self, session, key) -> dict | None:
        """Read checkpoint data. Returns None if not found."""

    def get_checkpoint_full(self, session, key) -> dict | None:
        """Read checkpoint with metadata."""
```

Implementations: `DynamoDBDataStore`, `S3DataStore` (extensible to Redis, Firestore, HTTP).

### 6.5 FaaS Invocation Backend

```python
class InvocationBackend(ABC):
    def invoke(self, function_name: str, payload: dict) -> None:
        """Asynchronously invoke a function with the given payload."""
```

Implementations:

- `AWSLambdaBackend`: Uses `boto3.client('lambda').invoke(InvocationType='Event')`
- `GCloudFunctionBackend`: Publishes to Pub/Sub topic
- Cross-platform: Runtime reads `Platform` field from config to select backend

### 6.6 Exactly-Once Guarantee

The system guarantees exactly-once execution semantics via:

1. **Checkpoint-before-invoke**: Each function checkpoints its output BEFORE invoking the next
2. **Atomic writes**: DynamoDB `PutItem` with `ConditionExpression: attribute_not_exists(pk)`
3. **Idempotent check**: Each function checks for existing checkpoint before executing
4. **At-least-once invocation**: FaaS may retry; duplicate invocations are caught by checkpoint

---

## 7. Denotational Semantics Sketch

Here's an informal sketch of the
denotational semantics that could be formalized:

### 7.1 Domains

```
Value  = Number | String | Bool | Array(Value) | Object(str → Value) | Future(Value)
Env    = Identifier → Value
Config = FunctionName → UnumConfig
Store  = (Session × Key) → Value        (persistent checkpoint store)
```

### 7.2 Semantic Functions

```
⟦·⟧ : AST × Env → (Value × Config × Store)

⟦ number(n) ⟧ ρ                    = (n, ∅, ∅)
⟦ string(s) ⟧ ρ                    = (s, ∅, ∅)
⟦ id(x) ⟧ ρ                        = (ρ(x), ∅, ∅)
⟦ assign(id(x), e) ⟧ ρ             = let (v, c, s) = ⟦e⟧ρ in (v, c, s), ρ[x ↦ v]
⟦ invocation(f, args) ⟧ ρ          = (invoke(f, ⟦args⟧ρ), {f ↦ config(f)}, ∅)
⟦ block_expr([s₁, ..., sₙ]) ⟧ ρ   = foldl (⟦sᵢ⟧) over ρ₀ = ρ
⟦ return(e) ⟧ ρ                    = ⟦e⟧ρ
⟦ map_expr(x, iter, body) ⟧ ρ      = let vs = ⟦iter⟧ρ in
                                       ([⟦body⟧(ρ[x ↦ vᵢ]) | vᵢ ∈ vs],
                                        {MapEntry ↦ {Next: bodyFunc, Map}},
                                        ∅)
⟦ collect(src, target) ⟧ ρ         = let srcs = ⟦src⟧ρ in
                                       (⟦target⟧ρ,
                                        {srcᵢ ↦ {Next: target, Fan-in} | srcᵢ ∈ srcs},
                                        ∅)
⟦ parallel([e₁,...,eₙ]) ⟧ ρ       = ([⟦e₁⟧ρ ∥ ... ∥ ⟦eₙ⟧ρ],
                                        {prev ↦ {Next: [f₁,...,fₙ], Scalar}},
                                        ∅)
⟦ if_expr(c, t, e) ⟧ ρ            = if ⟦c⟧ρ then ⟦t⟧ρ else ⟦e⟧ρ
⟦ lambda(x, body) ⟧ ρ             = (λv. ⟦body⟧(ρ[x ↦ v]), ∅, ∅)
⟦ platform(p, e) ⟧ ρ              = let (v, c, s) = ⟦e⟧ρ in
                                       (v, c[f.Platform ↦ p], s)
```

### 7.3 Key Properties for Formal Verification

1. **Determinism**: For a given input, the workflow always produces the same output
   (modulo function execution non-determinism)

2. **Confluence**: The order of parallel branch completion doesn't affect the final result
   (fan-in collects all values regardless of arrival order)

3. **Progress**: Every non-terminal function eventually invokes its successor(s)
   (guaranteed by the checkpoint-then-invoke protocol)

4. **Exactly-once**: Each function in a workflow execution runs at most once
   (guaranteed by atomic checkpoint checking)

5. **Type preservation** (planned): If `F : A → B` and `G : B → C` and the workflow
   chains `F → G`, then the runtime payload at `G`'s input is of type `B`

### 7.4 Operational Semantics: Fan-in

The fan-in coordination can be modeled as a barrier:

```
            Fan-in(sources: [s₁, ..., sₙ], target: t)

    s₁ completes → checkpoint(s₁.result)  → increment_counter
    s₂ completes → checkpoint(s₂.result)  → increment_counter
    ...
    sₙ completes → checkpoint(sₙ.result)  → increment_counter
                                              │
                                              ▼
                                    counter == n? ──yes──→ invoke(t, [s₁.result, ..., sₙ.result])
                                         │
                                        no ──→ (done, wait for others)
```

The atomic increment ensures exactly one source triggers the target invocation.

---

## 8. File Reference

### Implementation Files

| File                                  | Purpose                 | Key Functions                                          |
| ------------------------------------- | ----------------------- | ------------------------------------------------------ |
| `unum/unum-cli/dagl_compiler.py`      | Tokenizer + Parser      | `tokenize()`, `Parser.parse()`, `compile_dagl()`       |
| `unum/unum-cli/dagl_to_unum_ir.py`    | AST → Unum IR generator | `WorkflowGraph.compile()`, `compile_dagl_to_unum_ir()` |
| `unum/unum-cli/unum-cli.py`           | CLI entry point         | `compile_dagl_workflow()`, `compile_workflow()`        |
| `unum/runtime/unum.py`                | Runtime orchestration   | `Unum.egress()`, `Unum.ingress()`                      |
| `unum/runtime/ds.py`                  | Datastore abstraction   | `checkpoint_data()`, `get_checkpoint()`                |
| `unum/runtime/faas_invoke_backend.py` | Platform invocation     | `InvocationBackend.invoke()`                           |

### Test Files

| File                                    | Tests                                                                         |
| --------------------------------------- | ----------------------------------------------------------------------------- |
| `unum/unum-cli/test_dagl_compiler.py`   | Tokenizer (8) + Parser (31) + Extensions (9) + Workflows (18) = 52 tests      |
| `unum/unum-cli/test_dagl_to_unum_ir.py` | Hello World + Chain + Parallel + Wordcount + Platform + Directives = 52 tests |

### Example Workflows

| File                                        | Pattern                                  |
| ------------------------------------------- | ---------------------------------------- |
| `unum/unum-cli/examples/wordcount.dag`      | Map-Reduce (map → fan-in → map → fan-in) |
| `unum/unum-cli/examples/hello-world.dag`    | Simple chain (A → B)                     |
| `unum/unum-cli/examples/cross-platform.dag` | Cross-platform annotations (@aws, @gcp)  |
| `unum/unum-cli/examples/parallel.dag`       | Parallel fan-out + fan-in (collect)      |

---

## 9. Examples

### 9.1 Hello World

**Source** (`hello-world.dag`):

```dagl
let greeting = Hello(name: "World")
return World(msg: greeting)
```

**AST**:

```json
{
  "data": "block_expr",
  "children": [
    {
      "data": "assign",
      "children": [
        { "data": "id", "children": ["greeting"] },
        {
          "data": "invocation",
          "children": [
            "Hello",
            {
              "data": "dict",
              "children": [
                {
                  "data": "pair",
                  "children": [
                    { "data": "id", "children": ["name"] },
                    { "data": "string", "children": ["World"] }
                  ]
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "data": "return",
      "children": [
        {
          "data": "invocation",
          "children": [
            "World",
            {
              "data": "dict",
              "children": [
                {
                  "data": "pair",
                  "children": [
                    { "data": "id", "children": ["msg"] },
                    { "data": "id", "children": ["greeting"] }
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

**Unum IR**:

```
Hello/unum_config.json:
  {"Name": "Hello", "Start": true, "Checkpoint": true, "Debug": true,
   "Next": {"Name": "World", "InputType": "Scalar"}}

World/unum_config.json:
  {"Name": "World", "Start": false, "Checkpoint": true, "Debug": true}
```

**Runtime execution**:

```
1. Trigger workflow with input {"name": "World"}
2. Hello receives: {Data: {Source: "http", Value: {"name": "World"}}, Session: "abc-123"}
3. Hello executes, outputs: {"message": "Hello, World!"}
4. Hello checkpoints, then invokes World with:
   {Data: {Source: "http", Value: {"message": "Hello, World!"}}, Session: "abc-123"}
5. World executes, outputs final result
6. World checkpoints (terminal — no next invocation)
```

### 9.2 Wordcount (Map-Reduce)

**Source** (`wordcount.dag`):

```dagl
@workflow("wordcount")
@checkpoint(true)
@eager(true)

let counts = map chunk in input {
    return Mapper(chunk)
}
let partitions = Partition(counts)
let reduced = map p in partitions {
    return Reducer(p)
}
return Summary(reduced)
```

**Compiled IR** (5 functions):

```
UnumMap0:  Start=true,  Next={Mapper, Map}
Mapper:    Start=false, Next={Partition, Fan-in: ["Mapper-unumIndex-*"]}, Pop
Partition: Start=false, Next={Reducer, Map}
Reducer:   Start=false, Next={Summary, Fan-in: ["Reducer-unumIndex-*"]}, Pop
Summary:   Start=false, (terminal)
```

**Runtime execution** (input = ["chunk1", "chunk2", "chunk3"]):

```
1. UnumMap0 receives input array, fans out:
   → Mapper-UnumIndex-0 (chunk1)
   → Mapper-UnumIndex-1 (chunk2)
   → Mapper-UnumIndex-2 (chunk3)

2. Each Mapper executes independently, checkpoints result
   Last Mapper to complete triggers fan-in → invokes Partition

3. Partition receives [mapper0.result, mapper1.result, mapper2.result]
   Partitions data, outputs array → fans out to Reducers

4. Each Reducer executes, last one triggers fan-in → invokes Summary

5. Summary receives all reducer results, produces final output
```

### 9.3 Cross-Platform

**Source** (`cross-platform.dag`):

```dagl
@workflow("cross-platform-pipeline")
@checkpoint(true)

let data = @aws Ingest(rawInput)
let processed = @gcp Transform(data)
return @aws Store(processed)
```

**Compiled IR**:

```
Ingest:    Start=true,  Next={Transform, Scalar}, Platform="aws"
Transform: Start=false, Next={Store, Scalar},     Platform="gcp"
Store:     Start=false, (terminal),                Platform="aws"
```

**Runtime**: The `Platform` field tells the invocation backend which cloud provider to use when invoking the next function. `Ingest` on AWS Lambda invokes `Transform` on GCP Cloud Functions, which invokes `Store` back on AWS Lambda.

---

## CLI Usage

```bash
# Compile DAGL-U workflow
unum-cli compile -p dagl -w workflow.dag

# Compile with existing template (merges directives)
unum-cli compile -p dagl -w workflow.dag -t unum-template.yaml

# Compile Step Functions (existing path, unchanged)
unum-cli compile -p step-functions -w workflow.json -t unum-template.yaml

# Build, deploy (unchanged — works with generated configs)
unum-cli build -p aws
unum-cli deploy -p aws

# Invoke and wait for result
unum-cli invoke --wait
```

---

## 10. Step-by-Step: Running hello-world-dagl

This walkthrough shows how to go from source files to a running workflow on AWS,
using the `hello-world-dagl` example from `unum-appstore/`.

### 10.1 Prerequisites

- Python 3.11+
- AWS SAM CLI (`sam --version`)
- AWS credentials configured (`AWS_PROFILE` environment variable set)
- A DynamoDB table named `unum-intermediate-datastore` (used for checkpointing)

### 10.2 Source Files (what you write)

The starting directory contains **only user-authored files** — no generated artifacts:

```
hello-world-dagl/
├── workflow.dag          ← workflow definition (DAGL-U source)
├── input.json            ← sample input payload
├── Hello/
│   ├── app.py            ← user function code
│   ├── requirements.txt  ← Python dependencies
│   └── __init__.py
└── World/
    ├── app.py            ← user function code
    ├── requirements.txt  ← Python dependencies
    └── __init__.py
```

**workflow.dag** — the single source of truth for the workflow:
```dagl
@workflow("unum-hello-world-dagl")
@checkpoint(true)

let greeting = Hello()
return World(greeting)
```

**Hello/app.py**:
```python
def lambda_handler(event, context):
    return "Hello"
```

**World/app.py**:
```python
def lambda_handler(event, context):
    return f'{event} world!'
```

### 10.3 Step 1 — Compile

```bash
unum-cli compile -p dagl -w workflow.dag
```

**What it does**: Reads `workflow.dag`, tokenizes → parses → generates IR.

**What it creates**:
- `Hello/unum_config.json` — tells Hello it is the start function and its next is World
- `World/unum_config.json` — tells World it is the terminal function
- `unum-template.yaml` — workflow-level metadata (app name, datastore, platform)

**Output**:
```
Compiling DAGL-U workflow...
Loaded DAGL-U source: workflow.dag

Generating unum_config.json files...
  ✓ Hello/unum_config.json
  ✓ World/unum_config.json
  ✓ unum-template.yaml

DAGL-U compilation succeeded! Generated 2 unum_config.json files.

Workflow Summary:
  Start functions: Hello
  End functions: World
  Total functions: 2
```

After this step, the generated `Hello/unum_config.json` contains:
```json
{
  "Name": "Hello",
  "Start": true,
  "Checkpoint": true,
  "Debug": true,
  "Next": {
    "Name": "World",
    "InputType": "Scalar"
  }
}
```

### 10.4 Step 2 — Generate Platform Template

```bash
unum-cli template
```

**What it does**: Reads `unum-template.yaml` and generates an AWS SAM `template.yaml`.

**What it creates**:
- `template.yaml` — AWS SAM template with `HelloFunction` and `WorldFunction` resources,
  IAM policies, environment variables, and output ARN references.

### 10.5 Step 3 — Build

```bash
unum-cli build -p aws
```

**What it does**:
1. Copies the Unum runtime files (`unum.py`, `ds.py`, `main.py`, `faas_invoke_backend.py`)
   into a `common/` directory
2. Copies those runtime files into each function directory (`Hello/`, `World/`)
3. Runs `sam build -t template.yaml` to produce `.aws-sam/build/` artifacts

**What it creates**:
- `common/` — Unum runtime source (shared)
- `.aws-sam/build/HelloFunction/` — packaged Hello function with runtime + user code + config
- `.aws-sam/build/WorldFunction/` — packaged World function

### 10.6 Step 4 — Deploy

```bash
unum-cli deploy -p aws
```

**What it does**:
1. First deployment: runs `sam deploy` to create the CloudFormation stack and Lambda functions
2. Extracts deployed Lambda ARNs from CloudFormation outputs
3. Writes `function-arn.yaml` mapping logical names → ARNs
4. Updates `unum_config.json` in `.aws-sam/build/` to replace function names with ARNs
5. Runs `sam deploy` again with ARN-resolved configs

**What it creates**:
- `function-arn.yaml` — auto-generated mapping, e.g.:
  ```yaml
  Hello: arn:aws:lambda:eu-central-1:123456789:function:unum-hello-world-dagl-HelloFunction-abc123
  World: arn:aws:lambda:eu-central-1:123456789:function:unum-hello-world-dagl-WorldFunction-def456
  ```
- Two Lambda functions on AWS
- A CloudFormation stack named `unum-hello-world-dagl`

### 10.7 Step 5 — Invoke and Get the Result

```bash
unum-cli invoke --wait
```

**What it does**:
1. Reads `unum-template.yaml` to find the start function (`Hello`)
2. Reads `function-arn.yaml` to get its Lambda ARN
3. Generates a unique session ID
4. Sends the Unum payload `{Data: {Source: "http", Value: {}}, Session: "<uuid>"}`
5. Polls DynamoDB for the terminal function's (`World`) checkpoint until it appears

**Output**:
```
Workflow:  unum-hello-world-dagl
Start:     Hello
End:       World
Session:   3ded1db7-85a8-4bd1-95c4-8151b42420e6
Mode:      sync + wait

Start function returned: 200
Waiting for workflow to complete (timeout: 60s)...

Workflow completed in ~1s

World result:
"Hello world!"
```

### 10.8 What Happened at Runtime

```
┌─────────────────────────────────────────────────────────────────┐
│  invoke --wait                                                  │
│  ─────────────                                                  │
│  1. Calls Hello Lambda with:                                    │
│     {Data: {Source:"http", Value:{}}, Session:"3ded..."}        │
│                                                                 │
│  Hello Lambda (main.py → unum.py → app.py)                     │
│  ──────────────────────────────────────────                     │
│  2. unum.ingress() extracts {} as user input                   │
│  3. app.lambda_handler({}, ctx) returns "Hello"                 │
│  4. unum.checkpoint("3ded.../Hello-output", "Hello")            │
│  5. unum.egress() reads config → Next: World, Scalar           │
│  6. Invokes World Lambda with:                                  │
│     {Data: {Source:"http", Value:"Hello"}, Session:"3ded..."}   │
│                                                                 │
│  World Lambda                                                   │
│  ────────────                                                   │
│  7. unum.ingress() extracts "Hello" as user input               │
│  8. app.lambda_handler("Hello", ctx) returns "Hello world!"     │
│  9. unum.checkpoint("3ded.../World-output", "Hello world!")     │
│  10. No Next → terminal, done                                   │
│                                                                 │
│  Back in invoke --wait                                          │
│  ─────────────────────                                          │
│  11. Polls DynamoDB for key "3ded.../World-output"              │
│  12. Found! Returns "Hello world!"                              │
└─────────────────────────────────────────────────────────────────┘
```

### 10.9 Summary of All Commands

```bash
cd unum-appstore/hello-world-dagl

# Compile:  workflow.dag → unum_config.json + unum-template.yaml
unum-cli compile -p dagl -w workflow.dag

# Template: unum-template.yaml → template.yaml (SAM)
unum-cli template

# Build:    template.yaml → .aws-sam/build/ (with runtime files)
unum-cli build -p aws

# Deploy:   .aws-sam/build/ → AWS Lambda + function-arn.yaml
unum-cli deploy -p aws

# Invoke:   trigger workflow, wait for final result
unum-cli invoke --wait
```

Where `unum-cli` is `python <path-to>/unum/unum-cli/unum-cli.py`.

### 10.10 Directory After Full Pipeline

```
hello-world-dagl/
├── workflow.dag              ← authored
├── input.json                ← authored
├── unum-template.yaml        ← generated (compile)
├── template.yaml             ← generated (template)
├── function-arn.yaml         ← generated (deploy)
├── common/                   ← generated (build) — runtime files
├── .aws-sam/                 ← generated (build) — SAM artifacts
├── Hello/
│   ├── app.py                ← authored
│   ├── requirements.txt      ← authored
│   ├── __init__.py           ← authored
│   ├── unum_config.json      ← generated (compile)
│   ├── main.py               ← generated (build) — Lambda entry point
│   ├── unum.py               ← generated (build) — runtime
│   ├── ds.py                 ← generated (build) — datastore driver
│   └── faas_invoke_backend.py← generated (build) — invocation backend
└── World/
    ├── app.py                ← authored
    ├── requirements.txt      ← authored
    ├── __init__.py           ← authored
    ├── unum_config.json      ← generated (compile)
    ├── main.py               ← generated (build)
    ├── unum.py               ← generated (build)
    ├── ds.py                 ← generated (build)
    └── faas_invoke_backend.py← generated (build)
```
