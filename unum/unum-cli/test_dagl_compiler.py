"""Tests for DAGL-U compiler (tokenizer + parser)."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from dagl_compiler import tokenize, compile_dagl, node


def assert_eq(label, actual, expected):
    if actual != expected:
        print(f"  FAIL: {label}")
        print(f"    expected: {json.dumps(expected, indent=2)}")
        print(f"    actual:   {json.dumps(actual, indent=2)}")
        return False
    print(f"  PASS: {label}")
    return True


def test_tokenizer():
    print("\n=== Tokenizer Tests ===")
    passed = 0

    # Basic tokens
    tokens = tokenize('let x = 42')
    types = [t.type for t in tokens]
    passed += assert_eq("let assignment tokens", types, ["LET", "IDENTIFIER", "ASSIGN", "NUMBER"])

    # Action path
    tokens = tokenize('/_/hello(name: "Alice")')
    types = [t.type for t in tokens]
    passed += assert_eq("action path tokens", types, ["ACTION_PATH", "LPAREN", "IDENTIFIER", "COLON", "STRING", "RPAREN"])

    # Directive
    tokens = tokenize('@workflow("test")')
    types = [t.type for t in tokens]
    passed += assert_eq("directive tokens", types, ["DIRECTIVE", "LPAREN", "STRING", "RPAREN"])

    # Keywords
    tokens = tokenize('map x in items { return x }')
    types = [t.type for t in tokens]
    passed += assert_eq("map keyword tokens", types, ["MAP", "IDENTIFIER", "IN", "IDENTIFIER", "LBRACE", "RETURN", "IDENTIFIER", "RBRACE"])

    # collect/into
    tokens = tokenize('collect results into Aggregator()')
    types = [t.type for t in tokens]
    passed += assert_eq("collect/into tokens", types, ["COLLECT", "IDENTIFIER", "INTO", "IDENTIFIER", "LPAREN", "RPAREN"])

    # Plain function name (DAGL-U style)
    tokens = tokenize('Mapper(chunk)')
    types = [t.type for t in tokens]
    passed += assert_eq("plain function tokens", types, ["IDENTIFIER", "LPAREN", "IDENTIFIER", "RPAREN"])

    # Comments skipped
    tokens = tokenize('// this is a comment\nlet x = 1')
    types = [t.type for t in tokens]
    passed += assert_eq("comment skipped", types, ["LET", "IDENTIFIER", "ASSIGN", "NUMBER"])

    # Lambda
    tokens = tokenize('\\x -> x * 2')
    types = [t.type for t in tokens]
    passed += assert_eq("lambda tokens", types, ["LAMBDA", "IDENTIFIER", "ARROW", "IDENTIFIER", "MULT", "NUMBER"])

    print(f"\nTokenizer: {passed}/8 passed")
    return passed


def test_parser_basic():
    print("\n=== Parser Basic Tests ===")
    passed = 0

    # Number literal
    ast = compile_dagl("42")
    passed += assert_eq("number literal", ast, node("block_expr", [node("number", [42.0])]))

    # String literal
    ast = compile_dagl('"hello"')
    passed += assert_eq("string literal", ast, node("block_expr", [node("string", ["hello"])]))

    # Let assignment
    ast = compile_dagl("let x = 42")
    passed += assert_eq("let assignment", ast,
        node("block_expr", [
            node("assign", [node("id", ["x"]), node("number", [42.0])])
        ]))

    # Return
    ast = compile_dagl("let x = 10\nreturn x")
    passed += assert_eq("let + return", ast,
        node("block_expr", [
            node("assign", [node("id", ["x"]), node("number", [10.0])]),
            node("return", [node("id", ["x"])])
        ]))

    # Binary operation
    ast = compile_dagl("return 1 + 2")
    passed += assert_eq("binop", ast,
        node("block_expr", [
            node("return", [node("binop", [node("number", [1.0]), "+", node("number", [2.0])])])
        ]))

    print(f"\nParser basic: {passed}/5 passed")
    return passed


def test_parser_invocations():
    print("\n=== Parser Invocation Tests ===")
    passed = 0

    # Action path invocation (legacy DAGL)
    ast = compile_dagl('/_/hello(name: "Alice")')
    expected = node("block_expr", [
        node("invocation", ["/_/hello", node("dict", [
            node("pair", [node("id", ["name"]), node("string", ["Alice"])])
        ])])
    ])
    passed += assert_eq("action path invocation", ast, expected)

    # Plain function invocation (DAGL-U)
    ast = compile_dagl('Mapper(chunk)')
    expected = node("block_expr", [
        node("invocation", ["Mapper", node("list", [node("id", ["chunk"])])])
    ])
    passed += assert_eq("plain function invocation", ast, expected)

    # No-argument invocation
    ast = compile_dagl('Start()')
    expected = node("block_expr", [
        node("invocation", ["Start", node("dict", [])])
    ])
    passed += assert_eq("no-arg invocation", ast, expected)

    # Chain: let binding + invocation
    ast = compile_dagl('let x = FuncA()\nreturn FuncB(x)')
    expected = node("block_expr", [
        node("assign", [node("id", ["x"]), node("invocation", ["FuncA", node("dict", [])])]),
        node("return", [node("invocation", ["FuncB", node("list", [node("id", ["x"])])])])
    ])
    passed += assert_eq("chain invocation", ast, expected)

    # Nested invocation
    ast = compile_dagl('return World(msg: Hello(name: "Dani"))')
    expected = node("block_expr", [
        node("return", [
            node("invocation", ["World", node("dict", [
                node("pair", [node("id", ["msg"]),
                    node("invocation", ["Hello", node("dict", [
                        node("pair", [node("id", ["name"]), node("string", ["Dani"])])
                    ])])
                ])
            ])])
        ])
    ])
    passed += assert_eq("nested invocation", ast, expected)

    # Parallel auto-detection: multiple invocations → list
    ast = compile_dagl('FuncA()\nFuncB()\nFuncC()')
    expected = node("block_expr", [
        node("list", [
            node("invocation", ["FuncA", node("dict", [])]),
            node("invocation", ["FuncB", node("dict", [])]),
            node("invocation", ["FuncC", node("dict", [])]),
        ])
    ])
    passed += assert_eq("parallel auto-detection", ast, expected)

    print(f"\nParser invocations: {passed}/6 passed")
    return passed


def test_parser_control_flow():
    print("\n=== Parser Control Flow Tests ===")
    passed = 0

    # If-else
    ast = compile_dagl('if x > 5 { return "big" } else { return "small" }')
    assert ast["data"] == "block_expr"
    if_node = ast["children"][0]
    passed += assert_eq("if_expr node type", if_node["data"], "if_expr")
    passed += assert_eq("if_expr has 3 children", len(if_node["children"]), 3)

    # Map
    ast = compile_dagl('let r = map x in items { return x }')
    assign = ast["children"][0]
    map_node = assign["children"][1]
    passed += assert_eq("map_expr node type", map_node["data"], "map_expr")
    passed += assert_eq("map variable", map_node["children"][0], "x")

    # Lambda
    ast = compile_dagl('let f = \\x -> x * 2')
    assign = ast["children"][0]
    lam = assign["children"][1]
    passed += assert_eq("lambda node type", lam["data"], "lambda")
    passed += assert_eq("lambda param", lam["children"][0], "x")

    print(f"\nParser control flow: {passed}/6 passed")
    return passed


def test_daglu_extensions():
    print("\n=== DAGL-U Extension Tests ===")
    passed = 0

    # Directive
    ast = compile_dagl('@workflow("wordcount")\n@checkpoint(true)\nlet x = 1\nreturn x')
    passed += assert_eq("directives parsed", ast["data"], "block_expr")
    d1 = ast["children"][0]
    passed += assert_eq("first directive", d1, node("directive", ["workflow", node("string", ["wordcount"])]))
    d2 = ast["children"][1]
    passed += assert_eq("second directive", d2, node("directive", ["checkpoint", node("id", ["true"])]))

    # Collect
    ast = compile_dagl('let results = [BranchA(), BranchB()]\ncollect results into Aggregator()')
    collect_node = ast["children"][1]
    passed += assert_eq("collect node type", collect_node["data"], "collect")
    passed += assert_eq("collect target", collect_node["children"][1]["data"], "invocation")

    # Parallel block
    ast = compile_dagl('let r = parallel { FuncA() FuncB() FuncC() }')
    assign = ast["children"][0]
    par = assign["children"][1]
    passed += assert_eq("parallel node type", par["data"], "parallel")
    passed += assert_eq("parallel has 3 children", len(par["children"]), 3)

    # Platform annotation
    ast = compile_dagl('let x = @aws Ingest(data)')
    assign = ast["children"][0]
    plat = assign["children"][1]
    passed += assert_eq("platform node type", plat["data"], "platform")
    passed += assert_eq("platform name", plat["children"][0], "aws")

    print(f"\nDAGL-U extensions: {passed}/8 passed")
    return passed


def test_wordcount_workflow():
    print("\n=== Wordcount Workflow Test ===")
    source = '''
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
'''
    ast = compile_dagl(source)
    passed = 0

    passed += assert_eq("top-level block_expr", ast["data"], "block_expr")

    # 3 directives + 4 statements = 7 children
    children = ast["children"]
    passed += assert_eq("total children", len(children), 7)

    # Directives
    passed += assert_eq("directive 1", children[0]["data"], "directive")
    passed += assert_eq("directive 1 name", children[0]["children"][0], "workflow")

    # First map (counts = map chunk in input { Mapper(chunk) })
    assign1 = children[3]
    passed += assert_eq("first assign", assign1["data"], "assign")
    passed += assert_eq("first assign var", assign1["children"][0]["children"][0], "counts")
    map1 = assign1["children"][1]
    passed += assert_eq("first map", map1["data"], "map_expr")
    passed += assert_eq("first map var", map1["children"][0], "chunk")

    # Partition(counts)
    assign2 = children[4]
    passed += assert_eq("partition assign", assign2["data"], "assign")
    invoke = assign2["children"][1]
    passed += assert_eq("partition invocation", invoke["data"], "invocation")
    passed += assert_eq("partition func name", invoke["children"][0], "Partition")

    # return Summary(reduced)
    ret = children[6]
    passed += assert_eq("return node", ret["data"], "return")
    passed += assert_eq("summary invocation", ret["children"][0]["data"], "invocation")
    passed += assert_eq("summary func name", ret["children"][0]["children"][0], "Summary")

    print(f"\nWordcount workflow: {passed}/14 passed")
    return passed


def test_hello_world_workflow():
    print("\n=== Hello World Workflow Test ===")
    source = '''
let greeting = Hello(name: "World")
return World(msg: greeting)
'''
    ast = compile_dagl(source)
    passed = 0

    passed += assert_eq("block_expr", ast["data"], "block_expr")
    passed += assert_eq("2 statements", len(ast["children"]), 2)

    # Hello invocation
    assign = ast["children"][0]
    invoke = assign["children"][1]
    passed += assert_eq("Hello invocation", invoke["children"][0], "Hello")

    # World invocation
    ret = ast["children"][1]
    invoke2 = ret["children"][0]
    passed += assert_eq("World invocation", invoke2["children"][0], "World")

    print(f"\nHello World workflow: {passed}/4 passed")
    return passed


if __name__ == "__main__":
    total = 0
    total += test_tokenizer()
    total += test_parser_basic()
    total += test_parser_invocations()
    total += test_parser_control_flow()
    total += test_daglu_extensions()
    total += test_wordcount_workflow()
    total += test_hello_world_workflow()
    print(f"\n{'='*50}")
    print(f"TOTAL: {total}/51 passed")
    if total < 51:
        sys.exit(1)
    print("All tests passed!")
