"""DAGL Runtime Layer - Lambda Entry Point

This module replaces the user's Lambda handler. It:
1. Reads the user's original handler from DAGL_USER_HANDLER env var
2. Reads the orchestration config from DAGL_CONFIG env var
3. Reads the function ARN mapping from DAGL_FUNCTION_MAP env var
4. Wraps the user function with DAGL orchestration (checkpoint, continuation, fan-in)

Lambda Handler: dagl_runtime.main.handler
"""

import json
import os
import time
import sys
import importlib
import threading
import concurrent.futures

if os.environ.get('FAAS_PLATFORM') == 'gcloud':
    import base64

from dagl_runtime.unum import Unum
from dagl_runtime.faas_invoke_backend import InvocationBackend, _is_multi_platform_map

# ─── Dynamic user handler import ────────────────────────────────────────────────

def _import_user_handler():
    """Import the user's original handler from DAGL_USER_HANDLER env var.
    
    Format: "module.path.handler_function" (e.g., "app.handler", "index.handler")
    """
    handler_path = os.environ.get('DAGL_USER_HANDLER')
    if not handler_path:
        raise RuntimeError(
            "DAGL_USER_HANDLER env var not set. "
            "Set it to your original handler (e.g., 'app.handler')"
        )
    
    # Split into module path and function name
    parts = handler_path.rsplit('.', 1)
    if len(parts) != 2:
        raise RuntimeError(
            f"Invalid DAGL_USER_HANDLER format: '{handler_path}'. "
            f"Expected 'module.function' (e.g., 'app.handler')"
        )
    
    module_name, function_name = parts
    
    # Import the module
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise RuntimeError(
            f"Cannot import user handler module '{module_name}': {e}"
        )
    
    # Get the handler function
    try:
        handler_fn = getattr(module, function_name)
    except AttributeError:
        raise RuntimeError(
            f"Module '{module_name}' has no function '{function_name}'"
        )
    
    return handler_fn


# ─── Configuration from environment ─────────────────────────────────────────────

def _load_config():
    """Load unum config from DAGL_CONFIG env var (JSON string)."""
    config_str = os.environ.get('DAGL_CONFIG')
    if not config_str:
        raise RuntimeError(
            "DAGL_CONFIG env var not set. "
            "Run 'dagl deploy' to configure this function."
        )
    return json.loads(config_str)


def _load_function_map():
    """Load function name → ARN mapping from DAGL_FUNCTION_MAP env var."""
    map_str = os.environ.get('DAGL_FUNCTION_MAP')
    if not map_str:
        raise RuntimeError(
            "DAGL_FUNCTION_MAP env var not set. "
            "Run 'dagl deploy' to configure this function."
        )
    return json.loads(map_str)


# ─── Streaming support (optional) ───────────────────────────────────────────────

try:
    from dagl_runtime.unum_streaming import (
        get_streaming_output,
        clear_streaming_output,
        is_future,
        resolve_all_futures,
        LazyFutureDict,
        register_unum_context,
        unregister_unum_context,
        was_streaming_invoked
    )
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    def get_streaming_output(): return None
    def clear_streaming_output(): pass
    def register_unum_context(u, e): pass
    def unregister_unum_context(): pass
    def was_streaming_invoked(): return False


# ─── Initialize at import time (cold start) ─────────────────────────────────────

user_lambda = _import_user_handler()
config = _load_config()

platform = os.environ.get('FAAS_PLATFORM', 'aws')
ds_type = os.environ.get('UNUM_INTERMEDIARY_DATASTORE_TYPE', 'dynamodb')
ds_name = os.environ.get('UNUM_INTERMEDIARY_DATASTORE_NAME', 'unum-intermediate-datastore')
gc_enabled = os.environ.get('GC', 'false')
workflow_name = os.environ.get('WORKFLOW_NAME', '')

unum = Unum(config, ds_type, ds_name, platform, gc_enabled)


# ─── Ingress ─────────────────────────────────────────────────────────────────────

def ingress(event):
    """Extract user function input from the DAGL envelope."""
    
    if event["Data"]["Source"] == "http":
        if unum.entry_function == False:
            if unum.gc == True:
                unum.my_gc_tasks = event['GC']

        input_value = event["Data"]["Value"]
        
        # Streaming support: check for future references
        if STREAMING_AVAILABLE and isinstance(input_value, dict):
            is_streaming = input_value.get('__streaming__', False)
            has_futures = any(
                isinstance(v, dict) and v.get('__unum_future__')
                for v in input_value.values()
            )
            if is_streaming or has_futures:
                if '__streaming__' in input_value:
                    del input_value['__streaming__']
                return LazyFutureDict(input_value)
        
        return input_value
    else:
        # Fan-in: read inputs from datastore
        eager_fanin = event["Data"].get("EagerFanIn", {})
        
        if eager_fanin.get("Enabled", False):
            poll_interval = float(os.environ.get('UNUM_EAGER_POLL_INTERVAL', '0.1'))
            timeout = float(os.environ.get('UNUM_EAGER_TIMEOUT', '300'))
            ready_names = eager_fanin.get("Ready", [])
            total = eager_fanin.get("TotalBranches", len(event["Data"]["Value"]))
            
            use_futures = os.environ.get('UNUM_FUTURE_BASED', 'false').lower() == 'true'
            
            if use_futures:
                future_inputs = unum.ds.create_future_inputs(
                    event["Session"], event["Data"]["Value"],
                    ready_names=ready_names, poll_interval=poll_interval, timeout=timeout
                )
                future_inputs.try_resolve_all()
                unum._future_inputs = future_inputs
                unum.fan_in_gc = True
                return future_inputs
            else:
                lazy_inputs = unum.ds.create_lazy_inputs(
                    event["Session"], event["Data"]["Value"],
                    poll_interval=poll_interval, timeout=timeout
                )
                lazy_inputs.try_resolve_all()
                unum._lazy_inputs = lazy_inputs
                unum.fan_in_gc = True
                return lazy_inputs
        
        # Standard fan-in
        ckpt_vals = unum.ds.read_input(event["Session"], event["Data"]["Value"])
        
        if unum.gc == True:
            gc_tasks = [ckpt["GC"] for ckpt in ckpt_vals]
            unum.my_gc_tasks = {k: v for t in gc_tasks for k, v in t.items()}
            unum.fan_in_gc = True

        return [ckpt["User"] for ckpt in ckpt_vals]


# ─── Egress ──────────────────────────────────────────────────────────────────────

def egress(user_function_output, event):
    """Post-processing: checkpoint and invoke continuations."""
    
    # Handle GC from lazy inputs
    if hasattr(unum, '_lazy_inputs') and unum._lazy_inputs is not None:
        try:
            unum.my_gc_tasks = unum._lazy_inputs.get_gc_tasks()
        except Exception:
            unum.my_gc_tasks = {}
        finally:
            unum._lazy_inputs = None

    # Streaming check
    streaming_output = get_streaming_output() if STREAMING_AVAILABLE else None
    streaming_invoked = was_streaming_invoked() if STREAMING_AVAILABLE else False
    
    if streaming_output is not None or streaming_invoked:
        clear_streaming_output()
        if unum.gc:
            gc = {unum.get_my_instance_name(event): unum.get_my_outgoing_edges(event, user_function_output)}
            checkpoint_data = {'GC': gc, 'User': json.dumps(user_function_output)}
        else:
            checkpoint_data = {'User': json.dumps(user_function_output)}
        unum.run_checkpoint(event, checkpoint_data)
        if unum.gc:
            unum.run_gc()
        unum.cleanup()
        return unum.curr_session, None

    # Build checkpoint data
    if unum.gc:
        gc = {unum.get_my_instance_name(event): unum.get_my_outgoing_edges(event, user_function_output)}
        checkpoint_data = {'GC': gc, 'User': json.dumps(user_function_output)}
    else:
        checkpoint_data = {'User': json.dumps(user_function_output)}

    # Early invoke optimization for scalar continuations
    early_invoke = os.environ.get('EARLY_INVOKE', 'false').lower() == 'true'
    has_only_scalar = unum.has_only_scalar_continuations() if hasattr(unum, 'has_only_scalar_continuations') else False
    
    session = None
    next_payload_metadata = None
    
    if early_invoke and has_only_scalar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            checkpoint_future = executor.submit(unum.run_checkpoint, event, checkpoint_data)
            session, next_payload_metadata = unum.run_continuation(event, user_function_output)
            checkpoint_future.result()
    else:
        ret = unum.run_checkpoint(event, checkpoint_data)
        if ret in (0, -2, None):
            session, next_payload_metadata = unum.run_continuation(event, user_function_output)

    session = unum.curr_session

    if unum.gc:
        unum.run_gc()
    unum.cleanup()

    return session, next_payload_metadata


# ─── Handler (Lambda entry point) ────────────────────────────────────────────────

def handler(event, context):
    """DAGL runtime handler — wraps user function with orchestration.
    
    Set Lambda Handler to: dagl_runtime.main.handler
    """
    
    unum.cleanup()

    # Reset metrics
    if hasattr(unum, 'ds') and unum.ds is not None and hasattr(unum.ds, 'reset_metrics'):
        unum.ds.reset_metrics()

    # Platform-specific event parsing
    if platform == 'gcloud':
        if 'data' in event:
            input_data = base64.b64decode(event['data']).decode('utf-8')
            input_data = json.loads(input_data)
    elif platform == 'aws':
        input_data = event

        # Handle non-DAGL events (no envelope)
        if 'Data' not in input_data:
            if unum.entry_function:
                # Entry function: auto-wrap into DAGL envelope
                import uuid

                if 'detail' in input_data and 'source' in input_data:
                    # EventBridge event
                    user_payload = input_data['detail']
                    session_id = input_data.get('id', str(uuid.uuid4()))
                    if unum.debug:
                        print(f'[DAGL] EventBridge event from {input_data.get("source")} / {input_data.get("detail-type")}')
                else:
                    # Raw JSON trigger
                    user_payload = input_data
                    session_id = str(uuid.uuid4())
                    if unum.debug:
                        print(f'[DAGL] Raw JSON event, session={session_id}')

                input_data = {
                    'Data': {'Source': 'http', 'Value': user_payload},
                    'Session': session_id
                }
            else:
                # Non-entry function invoked directly: passthrough to user handler
                if unum.debug:
                    print(f'[DAGL] Passthrough: non-DAGL event on non-entry function')
                return user_lambda(event, context)

    if unum.debug:
        print(f'[DAGL] Function: {unum.name}, Session: {input_data.get("Session", "?")}')

    # Check for existing checkpoint (idempotency)
    ckpt_ret = unum.get_checkpoint(input_data)
    
    if ckpt_ret is None:
        user_function_input = ingress(input_data)
        
        if STREAMING_AVAILABLE:
            register_unum_context(unum, input_data)
        
        try:
            user_function_output = user_lambda(user_function_input, context)
        finally:
            if STREAMING_AVAILABLE:
                unregister_unum_context()
        
        if unum.debug:
            print(f'[DAGL] Input: {user_function_input}')
            print(f'[DAGL] Output: {user_function_output}')
    else:
        user_function_output = ckpt_ret
        if unum.debug:
            print(f'[DAGL] Output from checkpoint: {user_function_output}')

    session, next_payload_metadata = egress(user_function_output, input_data)

    # Log metrics
    if hasattr(unum, 'ds') and unum.ds is not None and hasattr(unum.ds, 'log_metrics'):
        unum.ds.log_metrics()

    return user_function_output


# ─── GCP Cloud Functions v2 HTTP Handler ─────────────────────────────────────────

def gcp_http_handler(request):
    """DAGL runtime handler for GCP Cloud Functions v2 (HTTP trigger).
    
    GCP Cloud Functions v2 uses HTTP triggers by default.
    This function accepts POST requests with JSON body containing either:
      - A DAGL envelope: {"Data": {...}, "Session": "..."}
      - Raw JSON input (entry function only): {"text": "..."}
    
    Set GCP entry point to: dagl_runtime.main.gcp_http_handler
    
    @param request flask.Request object
    @return tuple (response_body, status_code, headers)
    """
    import uuid as uuid_mod

    # Only accept POST
    if request.method != 'POST':
        return ('Method not allowed', 405, {'Content-Type': 'text/plain'})
    
    # Parse JSON body
    try:
        event = request.get_json(force=True)
    except Exception as e:
        return (json.dumps({'error': f'Invalid JSON: {e}'}), 400, 
                {'Content-Type': 'application/json'})
    
    if not event:
        return (json.dumps({'error': 'Empty request body'}), 400,
                {'Content-Type': 'application/json'})

    # ── Core orchestration (same as Lambda handler) ──
    unum.cleanup()
    
    if hasattr(unum, 'ds') and unum.ds is not None and hasattr(unum.ds, 'reset_metrics'):
        unum.ds.reset_metrics()

    # GCP user functions take a Flask Request object (1 arg),
    # Lambda-style take (event, context). Detect and wrap accordingly.
    import inspect
    try:
        sig = inspect.signature(user_lambda)
        _user_fn_nargs = len(sig.parameters)
    except (ValueError, TypeError):
        _user_fn_nargs = 2  # default to Lambda-style

    class _MockRequest:
        """Lightweight mock of Flask Request for GCP user functions."""
        def __init__(self, data):
            self._data = data
            self.method = 'POST'
            self.content_type = 'application/json'
        def get_json(self, force=False, silent=False):
            return self._data

    def _call_user_fn(data):
        """Call user function with correct number of args.
        
        GCP functions return (body, status, headers) or a string/dict.
        Unwrap to get the actual data dict for DAGL orchestration.
        """
        if _user_fn_nargs <= 1:
            result = user_lambda(_MockRequest(data))
        else:
            result = user_lambda(data, None)
        
        # Unwrap GCP response tuple: (body, status, headers)
        if isinstance(result, tuple):
            body = result[0]
            if isinstance(body, str):
                try:
                    return json.loads(body)
                except (json.JSONDecodeError, TypeError):
                    return {"result": body}
            return body
        if isinstance(result, str):
            try:
                return json.loads(result)
            except (json.JSONDecodeError, TypeError):
                return {"result": result}
        return result

    # If event has DAGL envelope, use directly; otherwise wrap for entry function
    if 'Data' in event:
        input_data = event
    elif unum.entry_function:
        input_data = {
            'Data': {'Source': 'http', 'Value': event},
            'Session': str(uuid_mod.uuid4())
        }
    else:
        # Non-entry function, no DAGL envelope — passthrough
        result = _call_user_fn(event)
        return (json.dumps(result), 200, {'Content-Type': 'application/json'})

    if unum.debug:
        print(f'[DAGL GCP] Function: {unum.name}, Session: {input_data.get("Session", "?")}')

    # Check checkpoint (idempotency)
    ckpt_ret = unum.get_checkpoint(input_data)
    
    if ckpt_ret is None:
        user_function_input = ingress(input_data)
        
        if STREAMING_AVAILABLE:
            register_unum_context(unum, input_data)
        
        try:
            user_function_output = _call_user_fn(user_function_input)
        finally:
            if STREAMING_AVAILABLE:
                unregister_unum_context()
    else:
        user_function_output = ckpt_ret

    session, _ = egress(user_function_output, input_data)

    if hasattr(unum, 'ds') and unum.ds is not None and hasattr(unum.ds, 'log_metrics'):
        unum.ds.log_metrics()

    # Return 200 with output (for cross-platform callers that wait)
    return (json.dumps(user_function_output), 200, {'Content-Type': 'application/json'})
