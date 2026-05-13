"""DAGL-U Compiler: Tokenizer and Parser

Ports the DAGL (Dagular) DSL compiler from JavaScript to Python,
extended with Unum-specific constructs (@directives, collect/gather fan-in,
platform-agnostic function references).

Compilation pipeline:
    Source (.dag) → Tokenizer → Token[] → Parser → AST (dict)

AST node format:
    {"data": "<node_type>", "children": [...]}

Node types:
    block_expr  — sequential statements
    return      — return expression
    assign      — variable assignment (let x = expr)
    invocation  — function/action call: [name_str, args_dict_or_list]
    apply       — lambda/variable application: [fn_expr, arg_expr]
    if_expr     — conditional: [condition, then_block, else_block]
    map_expr    — map: [var_name_str, iterable_expr, body_expr]
    lambda      — lambda: [param_name_str, body_expr]
    collect     — fan-in: [source_list_expr, target_invocation]  (DAGL-U extension)
    parallel    — explicit parallel block: [expr, ...]           (DAGL-U extension)
    directive   — workflow metadata: [name_str, value_expr]      (DAGL-U extension)
    type_decl   — function type declaration                      (DAGL-U extension)
    list        — array literal / parallel invocations
    dict        — object literal / named params
    pair        — key-value pair: [id_node, value_node]
    binop       — binary op: [left_expr, op_str, right_expr]
    unop        — unary op: [op_str, operand_expr]
    index       — subscript: [expr, index_expr]
    id          — identifier / action path: [name_str]
    number      — numeric literal: [value_float]
    string      — string literal: [value_str]
"""

import re
from dataclasses import dataclass
from typing import Any


# ─── Token ──────────────────────────────────────────────────────────────────────

@dataclass
class Token:
    type: str
    value: str
    pos: int


# ─── Tokenizer ─────────────────────────────────────────────────────────────────

# Order matters: longer/more-specific patterns first.
TOKEN_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("WHITESPACE",   re.compile(r"\s+")),
    ("COMMENT",      re.compile(r"//[^\n]*")),
    ("COMMENT",      re.compile(r"/\*[\s\S]*?\*/")),

    # Keywords
    ("RETURN",       re.compile(r"return\b")),
    ("LET",          re.compile(r"let\b")),
    ("IF",           re.compile(r"if\b")),
    ("ELSE",         re.compile(r"else\b")),
    ("MAP",          re.compile(r"map\b")),
    ("IN",           re.compile(r"in\b")),
    ("COLLECT",      re.compile(r"collect\b")),     # DAGL-U extension
    ("INTO",         re.compile(r"into\b")),        # DAGL-U extension
    ("PARALLEL",     re.compile(r"parallel\b")),    # DAGL-U extension
    ("TRUE",         re.compile(r"true\b")),
    ("FALSE",        re.compile(r"false\b")),
    ("NOT",          re.compile(r"not\b")),
    ("AND",          re.compile(r"and\b")),
    ("OR",           re.compile(r"or\b")),

    # Literals
    ("NUMBER",       re.compile(r"\d+(\.\d+)?([eE][+-]?\d+)?")),
    ("STRING",       re.compile(r'"([^"\\]|\\.)*"')),
    ("STRING",       re.compile(r"'([^'\\]|\\.)*'")),

    # Directives (DAGL-U extension: @workflow, @checkpoint, @eager, @function, @aws, @gcp, @azure)
    ("DIRECTIVE",    re.compile(r"@[a-zA-Z_][a-zA-Z0-9_]*")),

    # Action paths  (legacy DAGL: /_/hello)
    ("ACTION_PATH",  re.compile(r"/[a-zA-Z0-9_/\-]+")),

    # Operators (longer first)
    ("LAMBDA",       re.compile(r"\\")),
    ("ARROW",        re.compile(r"->")),
    ("EQ",           re.compile(r"==")),
    ("NE",           re.compile(r"!=")),
    ("LE",           re.compile(r"<=")),
    ("GE",           re.compile(r">=")),
    ("LT",           re.compile(r"<")),
    ("GT",           re.compile(r">")),
    ("ASSIGN",       re.compile(r"=")),
    ("PLUS",         re.compile(r"\+")),
    ("MINUS",        re.compile(r"-")),
    ("MULT",         re.compile(r"\*")),
    ("DIV",          re.compile(r"/")),
    ("MOD",          re.compile(r"%")),

    # Delimiters
    ("LPAREN",       re.compile(r"\(")),
    ("RPAREN",       re.compile(r"\)")),
    ("LBRACE",       re.compile(r"\{")),
    ("RBRACE",       re.compile(r"\}")),
    ("LBRACKET",     re.compile(r"\[")),
    ("RBRACKET",     re.compile(r"\]")),
    ("COMMA",        re.compile(r",")),
    ("COLON",        re.compile(r":")),
    ("SEMICOLON",    re.compile(r";")),
    ("DOT",          re.compile(r"\.")),

    # Identifiers (last)
    ("IDENTIFIER",   re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")),
]


def tokenize(source: str) -> list[Token]:
    """Lexical analysis: source string → list of Tokens (skipping whitespace/comments)."""
    tokens: list[Token] = []
    pos = 0
    while pos < len(source):
        matched = False
        for token_type, pattern in TOKEN_PATTERNS:
            m = pattern.match(source, pos)
            if m:
                if token_type not in ("WHITESPACE", "COMMENT"):
                    tokens.append(Token(type=token_type, value=m.group(), pos=pos))
                pos = m.end()
                matched = True
                break
        if not matched:
            raise SyntaxError(f"Unexpected character '{source[pos]}' at position {pos}")
    return tokens


# ─── AST helpers ────────────────────────────────────────────────────────────────

def node(data: str, children: list | None = None) -> dict:
    """Create an AST node."""
    return {"data": data, "children": children or []}


# ─── Parser ─────────────────────────────────────────────────────────────────────

class Parser:
    """Recursive-descent parser with operator-precedence climbing.

    Grammar (informal, PEG-like):
        program        = directive* (block | multi_expr)
        directive      = DIRECTIVE '(' expression ')'
        block          = (let_assign | bare_assign | return | collect_stmt | expression)*
        let_assign     = 'let' IDENTIFIER '=' expression
        bare_assign    = IDENTIFIER '=' expression
        return         = 'return' expression
        collect_stmt   = 'collect' expression 'into' expression
        expression     = logical_or
        logical_or     = logical_and ('or' logical_and)*
        logical_and    = equality ('and' equality)*
        equality       = comparison (('==' | '!=') comparison)*
        comparison     = addition (('<' | '<=' | '>' | '>=') addition)*
        addition       = multiplication (('+' | '-') multiplication)*
        multiplication = unary (('*' | '/' | '%') unary)*
        unary          = ('not' | '-') unary | postfix
        postfix        = primary ('[' expression ']' | '(' arguments ')')*
        primary        = NUMBER | STRING | TRUE | FALSE
                       | ACTION_PATH | IDENTIFIER
                       | '[' (expression (',' expression)*)? ']'
                       | '{' (IDENTIFIER ':' expression (',' ...))? '}'
                       | '(' expression ')'
                       | if_expr | map_expr | parallel_expr | lambda_expr
        if_expr        = 'if' expression '{' block '}' ('else' '{' block '}')?
        map_expr       = 'map' IDENTIFIER 'in' expression ('{' block '}' | expression)
        parallel_expr  = 'parallel' '{' (expression)* '}'
        lambda_expr    = '\\' IDENTIFIER '->' expression
        arguments      = named_args | positional_args | empty
        named_args     = IDENTIFIER ':' expression (',' IDENTIFIER ':' expression)*
        positional_args= expression (',' expression)*
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.current = 0

    # ── Utilities ───────────────────────────────────────────────────────────

    def _peek(self) -> Token:
        if self.current < len(self.tokens):
            return self.tokens[self.current]
        return Token("EOF", "", -1)

    def _advance(self) -> Token:
        tok = self._peek()
        if tok.type != "EOF":
            self.current += 1
        return tok

    def _check(self, *types: str) -> bool:
        return self._peek().type in types

    def _match(self, *types: str) -> bool:
        if self._check(*types):
            self._advance()
            return True
        return False

    def _consume(self, tok_type: str, msg: str) -> Token:
        if self._check(tok_type):
            return self._advance()
        cur = self._peek()
        raise SyntaxError(f"{msg}. Expected {tok_type}, got {cur.type} ('{cur.value}') at pos {cur.pos}")

    def _previous(self) -> Token:
        return self.tokens[self.current - 1]

    def _at_end(self) -> bool:
        return self.current >= len(self.tokens)

    def _lookahead(self, offset: int = 1) -> Token:
        idx = self.current + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return Token("EOF", "", -1)

    # ── Program ─────────────────────────────────────────────────────────────

    def parse(self) -> dict:
        """Entry point. Returns the top-level AST."""
        if not self.tokens:
            return node("dict")

        # Parse directives first
        directives: list[dict] = []
        while self._check("DIRECTIVE"):
            directives.append(self._parse_directive())

        # Parse body
        if self._is_block_start():
            body = self._parse_block()
        else:
            exprs = self._parse_multiple_expressions()
            if len(exprs) == 1:
                body = self._make_execution_block(exprs)
            else:
                body = self._optimize_parallel(exprs)

        # If directives exist, wrap: block_expr with directives + body statements
        if directives:
            children = directives + (body["children"] if body["data"] == "block_expr" else [body])
            return node("block_expr", children)
        return body

    # ── Directives ──────────────────────────────────────────────────────────

    def _parse_directive(self) -> dict:
        tok = self._consume("DIRECTIVE", "Expected directive")
        name = tok.value[1:]  # strip '@'
        self._consume("LPAREN", f"Expected '(' after @{name}")
        # Support multi-arg directives (e.g., @import("FuncName", "arn:..."))
        args = []
        args.append(self._parse_expression())
        while self._match("COMMA"):
            args.append(self._parse_expression())
        self._consume("RPAREN", f"Expected ')' after @{name} value")
        if len(args) == 1:
            return node("directive", [name, args[0]])
        return node("directive", [name, node("list", args)])

    # ── Block detection ─────────────────────────────────────────────────────

    def _is_block_start(self) -> bool:
        look = self.current
        limit = min(look + 12, len(self.tokens))
        while look < limit:
            t = self.tokens[look]
            if t.type in ("LET", "RETURN", "COLLECT"):
                return True
            if t.type == "IDENTIFIER" and look + 1 < len(self.tokens) and self.tokens[look + 1].type == "ASSIGN":
                return True
            look += 1
        return False

    # ── Block ───────────────────────────────────────────────────────────────

    def _parse_block(self) -> dict:
        stmts: list[dict] = []
        while not self._at_end() and not self._check("RBRACE"):
            if self._check("LET"):
                stmts.append(self._parse_let())
            elif self._check("IDENTIFIER") and self._lookahead().type == "ASSIGN":
                stmts.append(self._parse_bare_assign())
            elif self._check("RETURN"):
                stmts.append(self._parse_return())
                break
            elif self._check("COLLECT"):
                stmts.append(self._parse_collect())
            else:
                stmts.append(self._parse_expression())
                if self._at_end() or self._check("RBRACE"):
                    break
        return node("block_expr", stmts)

    def _parse_multiple_expressions(self) -> list[dict]:
        exprs: list[dict] = []
        while not self._at_end():
            exprs.append(self._parse_expression())
            if self._at_end():
                break
        return exprs

    # ── Statements ──────────────────────────────────────────────────────────

    def _parse_let(self) -> dict:
        self._consume("LET", "Expected 'let'")
        name = self._consume("IDENTIFIER", "Expected variable name")
        self._consume("ASSIGN", "Expected '='")
        expr = self._parse_expression()
        return node("assign", [node("id", [name.value]), expr])

    def _parse_bare_assign(self) -> dict:
        name = self._consume("IDENTIFIER", "Expected variable name")
        self._consume("ASSIGN", "Expected '='")
        expr = self._parse_expression()
        return node("assign", [node("id", [name.value]), expr])

    def _parse_return(self) -> dict:
        self._consume("RETURN", "Expected 'return'")
        expr = self._parse_expression()
        return node("return", [expr])

    def _parse_collect(self) -> dict:
        """collect <source_expr> into <target_invocation>"""
        self._consume("COLLECT", "Expected 'collect'")
        source = self._parse_expression()
        self._consume("INTO", "Expected 'into' after collect source")
        target = self._parse_expression()
        return node("collect", [source, target])

    # ── Parallel optimization ───────────────────────────────────────────────

    @staticmethod
    def _is_invocation(expr: dict) -> bool:
        return isinstance(expr, dict) and expr.get("data") == "invocation"

    def _optimize_parallel(self, exprs: list[dict]) -> dict:
        if len(exprs) > 1 and all(self._is_invocation(e) for e in exprs):
            return node("block_expr", [node("list", exprs)])
        return self._make_execution_block(exprs)

    @staticmethod
    def _make_execution_block(exprs: list[dict]) -> dict:
        return node("block_expr", exprs)

    # ── Expression precedence climbing ──────────────────────────────────────

    def _parse_expression(self) -> dict:
        # Check for platform annotation: @aws, @gcp, @azure before an expression
        if self._check("DIRECTIVE"):
            dir_tok = self._peek()
            dir_name = dir_tok.value[1:]
            if dir_name in ("aws", "gcp", "azure"):
                self._advance()
                inner = self._parse_expression()
                # Wrap the expression in a platform annotation node
                return node("platform", [dir_name, inner])
        return self._parse_logical_or()

    def _parse_logical_or(self) -> dict:
        expr = self._parse_logical_and()
        while self._match("OR"):
            right = self._parse_logical_and()
            expr = node("binop", [expr, "or", right])
        return expr

    def _parse_logical_and(self) -> dict:
        expr = self._parse_equality()
        while self._match("AND"):
            right = self._parse_equality()
            expr = node("binop", [expr, "and", right])
        return expr

    def _parse_equality(self) -> dict:
        expr = self._parse_comparison()
        while self._match("EQ", "NE"):
            op = self._previous().value
            right = self._parse_comparison()
            expr = node("binop", [expr, op, right])
        return expr

    def _parse_comparison(self) -> dict:
        expr = self._parse_addition()
        while self._match("GT", "GE", "LT", "LE"):
            op = self._previous().value
            right = self._parse_addition()
            expr = node("binop", [expr, op, right])
        return expr

    def _parse_addition(self) -> dict:
        expr = self._parse_multiplication()
        while self._match("PLUS", "MINUS"):
            op = self._previous().value
            right = self._parse_multiplication()
            expr = node("binop", [expr, op, right])
        return expr

    def _parse_multiplication(self) -> dict:
        expr = self._parse_unary()
        while self._match("MULT", "DIV", "MOD"):
            op = self._previous().value
            right = self._parse_unary()
            expr = node("binop", [expr, op, right])
        return expr

    def _parse_unary(self) -> dict:
        if self._match("NOT", "MINUS"):
            op_tok = self._previous()
            op_str = "not" if op_tok.type == "NOT" else "-"
            right = self._parse_unary()
            return node("unop", [op_str, right])
        return self._parse_postfix()

    # ── Postfix: indexing & invocation ──────────────────────────────────────

    def _parse_postfix(self) -> dict:
        expr = self._parse_primary()
        while True:
            if self._match("LBRACKET"):
                idx = self._parse_expression()
                self._consume("RBRACKET", "Expected ']'")
                expr = node("index", [expr, idx])
            elif self._match("LPAREN"):
                args = self._parse_arguments()
                self._consume("RPAREN", "Expected ')'")
                # If expr is an action path or a plain identifier → invocation
                if expr.get("data") == "id":
                    name = expr["children"][0]
                    if isinstance(name, str) and name.startswith("/"):
                        expr = node("invocation", [name, args])
                    else:
                        # Plain function name → also invocation (DAGL-U: platform-agnostic)
                        expr = node("invocation", [name, args])
                else:
                    # Lambda or complex expression application
                    if args.get("data") == "list" and len(args.get("children", [])) == 1:
                        expr = node("apply", [expr, args["children"][0]])
                    else:
                        expr = node("apply", [expr, args])
            elif self._match("DOT"):
                # Dot access: expr.field → index(expr, "field")
                field = self._consume("IDENTIFIER", "Expected field name after '.'")
                expr = node("index", [expr, node("string", [field.value])])
            else:
                break
        return expr

    def _parse_arguments(self) -> dict:
        if self._check("RPAREN"):
            return node("dict", [])

        # Detect named args: IDENTIFIER COLON
        if self._check("IDENTIFIER") and self._lookahead().type == "COLON":
            pairs: list[dict] = []
            while True:
                key = self._consume("IDENTIFIER", "Expected parameter name")
                self._consume("COLON", "Expected ':'")
                val = self._parse_expression()
                pairs.append(node("pair", [node("id", [key.value]), val]))
                if not self._match("COMMA"):
                    break
            return node("dict", pairs)

        # Positional args
        args: list[dict] = []
        while True:
            args.append(self._parse_expression())
            if not self._match("COMMA"):
                break
        return node("list", args)

    # ── Primary ─────────────────────────────────────────────────────────────

    def _parse_primary(self) -> dict:
        if self._match("TRUE"):
            return node("id", ["true"])
        if self._match("FALSE"):
            return node("id", ["false"])
        if self._match("NUMBER"):
            return node("number", [float(self._previous().value)])
        if self._match("STRING"):
            raw = self._previous().value
            val = raw[1:-1]  # strip quotes
            # Handle escape sequences
            val = val.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
            val = val.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")
            return node("string", [val])
        if self._match("ACTION_PATH", "IDENTIFIER"):
            return node("id", [self._previous().value])
        if self._match("LBRACKET"):
            elements: list[dict] = []
            if not self._check("RBRACKET"):
                while True:
                    elements.append(self._parse_expression())
                    if not self._match("COMMA"):
                        break
            self._consume("RBRACKET", "Expected ']'")
            return node("list", elements)
        if self._match("LBRACE"):
            pairs: list[dict] = []
            if not self._check("RBRACE"):
                while True:
                    key = self._consume("IDENTIFIER", "Expected property name")
                    self._consume("COLON", "Expected ':'")
                    val = self._parse_expression()
                    pairs.append(node("pair", [node("id", [key.value]), val]))
                    if not self._match("COMMA"):
                        break
            self._consume("RBRACE", "Expected '}'")
            return node("dict", pairs)
        if self._match("LPAREN"):
            expr = self._parse_expression()
            self._consume("RPAREN", "Expected ')'")
            return expr
        if self._check("IF"):
            return self._parse_if()
        if self._check("MAP"):
            return self._parse_map()
        if self._check("PARALLEL"):
            return self._parse_parallel()
        if self._check("LAMBDA"):
            return self._parse_lambda()

        cur = self._peek()
        raise SyntaxError(f"Unexpected token: {cur.type} ('{cur.value}') at pos {cur.pos}")

    # ── Control structures ──────────────────────────────────────────────────

    def _parse_if(self) -> dict:
        self._consume("IF", "Expected 'if'")
        cond = self._parse_expression()
        self._consume("LBRACE", "Expected '{'")
        then_b = self._parse_block()
        self._consume("RBRACE", "Expected '}'")
        else_b = node("dict", [])
        if self._match("ELSE"):
            if self._check("IF"):
                # else if → parse if as the else branch
                else_b = node("block_expr", [self._parse_if()])
            else:
                self._consume("LBRACE", "Expected '{'")
                else_b = self._parse_block()
                self._consume("RBRACE", "Expected '}'")
        return node("if_expr", [cond, then_b, else_b])

    def _parse_map(self) -> dict:
        self._consume("MAP", "Expected 'map'")
        var = self._consume("IDENTIFIER", "Expected variable name")
        self._consume("IN", "Expected 'in'")
        iterable = self._parse_expression()
        if self._match("LBRACE"):
            body = self._parse_block()
            self._consume("RBRACE", "Expected '}'")
        else:
            body = self._parse_expression()
        return node("map_expr", [var.value, iterable, body])

    def _parse_parallel(self) -> dict:
        self._consume("PARALLEL", "Expected 'parallel'")
        self._consume("LBRACE", "Expected '{'")
        exprs: list[dict] = []
        while not self._check("RBRACE") and not self._at_end():
            exprs.append(self._parse_expression())
        self._consume("RBRACE", "Expected '}'")
        return node("parallel", exprs)

    def _parse_lambda(self) -> dict:
        self._consume("LAMBDA", "Expected '\\'")
        param = self._consume("IDENTIFIER", "Expected parameter name")
        self._consume("ARROW", "Expected '->'")
        body = self._parse_expression()
        return node("lambda", [param.value, body])


# ─── Public API ─────────────────────────────────────────────────────────────────

def compile_dagl(source: str) -> dict:
    """Compile DAGL-U source code to an AST dict.

    Args:
        source: DAGL-U source code string

    Returns:
        AST dict with {"data": ..., "children": [...]} structure

    Raises:
        SyntaxError: on tokenization or parsing errors
    """
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse()
