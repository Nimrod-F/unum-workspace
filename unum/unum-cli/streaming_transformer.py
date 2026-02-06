"""
AST Transformer for Partial Parameter Streaming

Analyzes function code to:
1. Identify return value construction (dict with field assignments)
2. Track which variables contribute to which fields
3. Find computation points for each field
4. Generate transformed code that:
   - Publishes fields as they're computed
   - Invokes next function after first field is ready
   - Receiver resolves futures on-demand
"""

import ast
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any


@dataclass
class FieldComputation:
    """Represents when and how a return field is computed"""
    field_name: str           # Name of the field in the return dict
    source_variable: str      # Variable name that holds the value
    computed_at_line: int     # Line number where the value is computed
    assignment_node: Any      # AST node of the assignment (for insertion point)


@dataclass
class StreamingAnalysis:
    """Result of analyzing a function for streaming potential"""
    can_stream: bool          # Whether streaming is possible
    reason: str               # Explanation of decision
    handler_name: str         # Name of the handler function
    return_line: int          # Line number of return statement
    return_node: Any          # AST node of return statement
    fields: List[FieldComputation]  # Ordered list of field computations


class StreamingAnalyzer:
    """
    Analyze a Python source file to determine streaming opportunities.
    
    Looks for patterns like:
        def lambda_handler(event, context):
            field1 = compute_field1()
            field2 = compute_field2()
            field3 = compute_field3()
            
            result = {
                "field1": field1,
                "field2": field2,
                "field3": field3
            }
            return result
    """
    
    def __init__(self):
        self.assignments: Dict[str, Tuple[int, ast.AST]] = {}  # var_name -> (line, node)
        self.return_node = None
        self.return_line = 0
        self.return_var = None  # Variable name if return is `return var`
        self.return_dict = None  # Dict contents if return is `return {...}`
        self.handler_node = None
        self.handler_name = ""
    
    def analyze(self, source: str) -> StreamingAnalysis:
        """
        Analyze source code for streaming opportunities.
        
        Args:
            source: Python source code as string
            
        Returns:
            StreamingAnalysis with results
        """
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return StreamingAnalysis(
                can_stream=False,
                reason=f"Syntax error: {e}",
                handler_name="",
                return_line=0,
                return_node=None,
                fields=[]
            )
        
        # Find the handler function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in ('lambda_handler', 'handler', 'main'):
                    self.handler_node = node
                    self.handler_name = node.name
                    break
        
        if not self.handler_node:
            return StreamingAnalysis(
                can_stream=False,
                reason="No handler function found (lambda_handler, handler, or main)",
                handler_name="",
                return_line=0,
                return_node=None,
                fields=[]
            )
        
        # Collect all assignments in the handler
        self._collect_assignments(self.handler_node)
        
        # Find the return statement
        self._find_return(self.handler_node)
        
        if not self.return_node:
            return StreamingAnalysis(
                can_stream=False,
                reason="No return statement found",
                handler_name=self.handler_name,
                return_line=0,
                return_node=None,
                fields=[]
            )
        
        # Analyze what's being returned
        fields = self._analyze_return()
        
        if len(fields) < 2:
            return StreamingAnalysis(
                can_stream=False,
                reason=f"Need at least 2 streamable fields, found {len(fields)}",
                handler_name=self.handler_name,
                return_line=self.return_line,
                return_node=self.return_node,
                fields=fields
            )
        
        # Check that fields are computed at different lines
        unique_lines = set(f.computed_at_line for f in fields)
        if len(unique_lines) < 2:
            return StreamingAnalysis(
                can_stream=False,
                reason="All fields computed at same line, no streaming benefit",
                handler_name=self.handler_name,
                return_line=self.return_line,
                return_node=self.return_node,
                fields=fields
            )
        
        return StreamingAnalysis(
            can_stream=True,
            reason="OK",
            handler_name=self.handler_name,
            return_line=self.return_line,
            return_node=self.return_node,
            fields=sorted(fields, key=lambda f: f.computed_at_line)
        )
    
    def _collect_assignments(self, func_node: ast.FunctionDef):
        """Collect all variable assignments in the function"""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.assignments[target.id] = (node.lineno, node)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                self.assignments[node.target.id] = (node.lineno, node)
    
    def _find_return(self, func_node: ast.FunctionDef):
        """Find the return statement in the function"""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return) and node.value:
                self.return_node = node
                self.return_line = node.lineno
                
                # Check if returning a variable or inline dict
                if isinstance(node.value, ast.Name):
                    self.return_var = node.value.id
                    # Look up the variable's assignment
                    if self.return_var in self.assignments:
                        assign_line, assign_node = self.assignments[self.return_var]
                        # Check if assigned to a dict
                        if isinstance(assign_node, ast.Assign):
                            if isinstance(assign_node.value, ast.Dict):
                                self.return_dict = assign_node.value
                elif isinstance(node.value, ast.Dict):
                    self.return_dict = node.value
                
                break  # Use first return statement
    
    def _analyze_return(self) -> List[FieldComputation]:
        """Analyze the return dict to find field computations"""
        fields = []
        
        if not self.return_dict:
            return fields
        
        for key_node, value_node in zip(self.return_dict.keys, self.return_dict.values):
            # Get field name
            if isinstance(key_node, ast.Constant):
                field_name = str(key_node.value)
            elif isinstance(key_node, ast.Str):  # Python 3.7 compatibility
                field_name = key_node.s
            else:
                continue
            
            # Get source variable and computation point
            if isinstance(value_node, ast.Name):
                var_name = value_node.id
                if var_name in self.assignments:
                    line, node = self.assignments[var_name]
                    fields.append(FieldComputation(
                        field_name=field_name,
                        source_variable=var_name,
                        computed_at_line=line,
                        assignment_node=node
                    ))
            elif isinstance(value_node, ast.Call):
                # Inline function call like {"result": compute_result()}
                # Can't easily stream this
                pass
        
        return fields


class StreamingTransformer:
    """
    Transform source code to enable partial parameter streaming.
    
    Injects code to:
    1. Import streaming runtime
    2. Create StreamingPublisher after session_id is available
    3. Call publisher.publish() after each field is computed
    4. Invoke next function after first field (with futures for pending)
    """
    
    def __init__(self, analysis: StreamingAnalysis, function_name: str):
        """
        Initialize transformer.
        
        Args:
            analysis: Result from StreamingAnalyzer
            function_name: Name of this function (for logging/debugging)
        """
        self.analysis = analysis
        self.function_name = function_name
    
    def transform(self, source: str) -> Tuple[str, List[str]]:
        """
        Transform source code to enable streaming.
        
        Args:
            source: Original Python source code
            
        Returns:
            Tuple of (transformed_source, list_of_messages)
        """
        messages = []
        lines = source.split('\n')
        
        # Track insertions (line_number -> code_to_insert_after)
        insertions: Dict[int, List[str]] = {}
        
        # 1. Add import at the very beginning (after any existing imports)
        import_line = self._find_import_insertion_point(lines)
        import_code = "from unum_streaming import StreamingPublisher, set_streaming_output"
        
        # 2. Find where to initialize the StreamingPublisher
        #    Should be after session_id is available (usually from event)
        init_line = self._find_init_point(lines)
        
        field_names_str = ", ".join(f'"{f.field_name}"' for f in self.analysis.fields)
        init_code = f'''
    # Streaming: Initialize publisher for incremental parameter streaming
    _streaming_session = event.get('Session', '') or str(id(event))
    _streaming_publisher = StreamingPublisher(
        session_id=_streaming_session,
        source_function="{self.function_name}",
        field_names=[{field_names_str}]
    )'''
        
        # 3. Add publish calls after each field computation
        for i, field in enumerate(self.analysis.fields):
            publish_code = f"    _streaming_publisher.publish('{field.field_name}', {field.source_variable})"
            
            # First field also triggers next function invocation
            if i == 0:
                publish_code += '''
    # Streaming: Signal to runtime to invoke next function early with futures
    if _streaming_publisher.should_invoke_next():
        _streaming_payload = _streaming_publisher.get_streaming_payload()
        # Store payload for runtime to pick up and invoke continuation
        set_streaming_output(_streaming_payload)
        _streaming_publisher.mark_next_invoked()'''
            
            if field.computed_at_line not in insertions:
                insertions[field.computed_at_line] = []
            insertions[field.computed_at_line].append(publish_code)
            
            messages.append(f"Stream field '{field.field_name}' after line {field.computed_at_line}")
        
        # Build new source
        new_lines = []
        
        # Add import at the top
        for i, line in enumerate(lines, 1):
            if i == import_line:
                new_lines.append(import_code)
            
            new_lines.append(line)
            
            if i == init_line:
                new_lines.append(init_code)
            
            if i in insertions:
                for code in insertions[i]:
                    new_lines.append(code)
        
        return '\n'.join(new_lines), messages
    
    def _find_import_insertion_point(self, lines: List[str]) -> int:
        """Find the line number after which to insert our import"""
        last_import = 0
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                last_import = i
            elif stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                if last_import > 0:
                    break
        
        return last_import if last_import > 0 else 1
    
    def _find_init_point(self, lines: List[str]) -> int:
        """
        Find where to initialize the StreamingPublisher.
        
        Should be:
        1. Inside the handler function
        2. After 'event' parameter is available
        3. Before first field computation
        """
        # Find handler function definition
        handler_line = 0
        for i, line in enumerate(lines, 1):
            if f'def {self.analysis.handler_name}' in line:
                handler_line = i
                break
        
        if handler_line == 0:
            return 1
        
        # First field computation line
        first_field_line = min(f.computed_at_line for f in self.analysis.fields)
        
        # Insert right after function definition (before first field)
        # Look for the first non-docstring, non-comment line after def
        for i in range(handler_line, first_field_line):
            line = lines[i - 1].strip() if i <= len(lines) else ""
            if line and not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''"):
                if not line.startswith('def '):
                    return i
        
        return handler_line


def analyze_file(filepath: str) -> StreamingAnalysis:
    """
    Convenience function to analyze a file for streaming.
    
    Args:
        filepath: Path to Python source file
        
    Returns:
        StreamingAnalysis result
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    analyzer = StreamingAnalyzer()
    return analyzer.analyze(source)


def transform_file(filepath: str, function_name: str) -> Tuple[str, List[str]]:
    """
    Convenience function to transform a file for streaming.
    
    Args:
        filepath: Path to Python source file
        function_name: Name of the function being transformed
        
    Returns:
        Tuple of (transformed_source, messages)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    analyzer = StreamingAnalyzer()
    analysis = analyzer.analyze(source)
    
    if not analysis.can_stream:
        return source, [f"Cannot stream: {analysis.reason}"]
    
    transformer = StreamingTransformer(analysis, function_name)
    return transformer.transform(source)


# For testing
if __name__ == "__main__":
    test_source = '''
import json
import time

def compute_field1(data):
    time.sleep(1)
    return {"mean": sum(data) / len(data)}

def compute_field2(data):
    time.sleep(1)
    return {"trend": data[-1] - data[0]}

def compute_field3(data):
    time.sleep(1)
    return {"count": len(data)}

def lambda_handler(event, context):
    data = event.get("data", [1, 2, 3, 4, 5])
    
    # Compute fields
    field1 = compute_field1(data)
    field2 = compute_field2(data)
    field3 = compute_field3(data)
    
    result = {
        "statistical": field1,
        "temporal": field2,
        "metadata": field3
    }
    
    return result
'''
    
    print("=== Analyzing ===")
    analyzer = StreamingAnalyzer()
    analysis = analyzer.analyze(test_source)
    
    print(f"Can stream: {analysis.can_stream}")
    print(f"Reason: {analysis.reason}")
    print(f"Handler: {analysis.handler_name}")
    print(f"Return line: {analysis.return_line}")
    print(f"Fields:")
    for f in analysis.fields:
        print(f"  - {f.field_name}: computed at line {f.computed_at_line} from {f.source_variable}")
    
    if analysis.can_stream:
        print("\n=== Transforming ===")
        transformer = StreamingTransformer(analysis, "TestFunction")
        new_source, messages = transformer.transform(test_source)
        
        print("Messages:")
        for msg in messages:
            print(f"  - {msg}")
        
        print("\n=== Transformed Source ===")
        print(new_source)
