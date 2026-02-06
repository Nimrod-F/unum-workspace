"""
Partial Parameter Streaming Runtime

Enables streaming of parameters from one function to another,
allowing the receiver to start processing before all params are ready.

Key concepts:
- Publisher: Function that computes parameters and publishes them incrementally
- Future: Reference to a parameter that will be available later
- Resolver: Mechanism to wait for and retrieve future values
"""

import json
import time
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

# Try to import datastore functions
try:
    from ds import write_intermediary, read_intermediary
except ImportError:
    # Fallback for testing
    _memory_store = {}
    def write_intermediary(key, value):
        _memory_store[key] = value
    def read_intermediary(key):
        return _memory_store.get(key)

# Configuration
STREAMING_POLL_INTERVAL = 0.1  # 100ms initial poll interval
STREAMING_MAX_POLL_INTERVAL = 1.0  # 1s max poll interval
STREAMING_TIMEOUT = 300  # 5 minutes max wait
STREAMING_DEBUG = os.environ.get('STREAMING_DEBUG', 'false').lower() == 'true'

def debug_log(msg: str):
    """Print debug message if debugging is enabled"""
    if STREAMING_DEBUG:
        print(f"[STREAMING] {msg}")

def get_session_id(event: dict) -> str:
    """Extract session ID from event"""
    return event.get('Session', '')

def make_streaming_key(session_id: str, source_function: str, field_name: str) -> str:
    """Create datastore key for a streaming field"""
    return f"streaming/{session_id}/{source_function}/{field_name}"

def make_future_ref(session_id: str, field_name: str, source_function: str) -> dict:
    """
    Create a future reference for a streaming field.
    
    A future reference is a placeholder that tells the receiver
    "this value will be available at this key in the datastore".
    """
    return {
        "__unum_future__": True,
        "session": session_id,
        "key": field_name,
        "source": source_function
    }

def is_future(value: Any) -> bool:
    """Check if a value is a future reference"""
    return isinstance(value, dict) and value.get("__unum_future__") == True

def publish_field(session_id: str, source_function: str, field_name: str, value: Any):
    """
    Publish a computed field to the datastore.
    
    Called by sender function after computing each field.
    The value is stored so the receiver can retrieve it.
    """
    key = make_streaming_key(session_id, source_function, field_name)
    
    # Store the value with metadata
    data = {
        "value": value,
        "timestamp": time.time(),
        "ready": True
    }
    
    try:
        write_intermediary(key, json.dumps(data, default=str))
        debug_log(f"Published field '{field_name}' to {key}")
    except Exception as e:
        print(f"[STREAMING] ERROR: Failed to publish field '{field_name}': {e}")
        raise

def resolve_future(future_ref: dict, timeout: float = STREAMING_TIMEOUT) -> Any:
    """
    Resolve a future reference by reading from datastore.
    
    Blocks until the value is available or timeout.
    Uses exponential backoff to avoid hammering the datastore.
    """
    if not is_future(future_ref):
        # Already a resolved value, return as-is
        return future_ref
    
    session_id = future_ref["session"]
    field_name = future_ref["key"]
    source_function = future_ref["source"]
    
    key = make_streaming_key(session_id, source_function, field_name)
    
    debug_log(f"Resolving future '{field_name}' from {key}")
    
    start_time = time.time()
    poll_interval = STREAMING_POLL_INTERVAL
    attempts = 0
    
    while True:
        attempts += 1
        try:
            data_str = read_intermediary(key)
            if data_str:
                data = json.loads(data_str)
                if data.get("ready"):
                    elapsed = time.time() - start_time
                    debug_log(f"Resolved '{field_name}' after {elapsed:.3f}s ({attempts} attempts)")
                    return data["value"]
        except json.JSONDecodeError as e:
            debug_log(f"JSON decode error for '{field_name}': {e}")
        except Exception as e:
            debug_log(f"Read error for '{field_name}': {e}")
        
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(
                f"Future '{field_name}' from '{source_function}' not resolved within {timeout}s "
                f"(session={session_id}, attempts={attempts})"
            )
        
        time.sleep(poll_interval)
        # Exponential backoff with cap
        poll_interval = min(poll_interval * 1.5, STREAMING_MAX_POLL_INTERVAL)

def resolve_all_futures(data: Any) -> Any:
    """
    Recursively resolve all futures in a data structure.
    
    Walks through dicts and lists, resolving any future references found.
    """
    if is_future(data):
        return resolve_future(data)
    
    if isinstance(data, dict):
        return {k: resolve_all_futures(v) for k, v in data.items()}
    
    if isinstance(data, list):
        return [resolve_all_futures(item) for item in data]
    
    return data

def create_streaming_payload(
    session_id: str,
    source_function: str,
    ready_fields: Dict[str, Any],
    pending_fields: List[str]
) -> dict:
    """
    Create a payload with mix of ready values and future references.
    
    Args:
        session_id: Current session ID
        source_function: Name of the function creating the payload
        ready_fields: Dict of field_name -> value for already computed fields
        pending_fields: List of field names that will be computed later
    
    Returns:
        Dict with ready values and future references for pending fields
    """
    payload = {}
    
    # Add ready fields as direct values
    for field_name, value in ready_fields.items():
        payload[field_name] = value
    
    # Add pending fields as future references
    for field_name in pending_fields:
        payload[field_name] = make_future_ref(session_id, field_name, source_function)
    
    return payload


class StreamingPublisher:
    """
    Context manager for streaming parameter publishing.
    
    Tracks which fields have been published and manages the
    lifecycle of streaming a result to the next function.
    """
    
    def __init__(self, session_id: str, source_function: str, field_names: List[str]):
        """
        Initialize streaming publisher.
        
        Args:
            session_id: Current session ID
            source_function: Name of this function
            field_names: List of all field names that will be published
        """
        self.session_id = session_id
        self.source_function = source_function
        self.field_names = field_names
        self.published_fields: Dict[str, Any] = {}
        self.next_function_invoked = False
        self._lock = threading.Lock()
        
    def publish(self, field_name: str, value: Any):
        """
        Publish a field value to the datastore.
        
        Args:
            field_name: Name of the field
            value: Computed value for the field
        """
        with self._lock:
            # Store to datastore
            publish_field(self.session_id, self.source_function, field_name, value)
            
            # Track locally
            self.published_fields[field_name] = value
            
            debug_log(f"Published {len(self.published_fields)}/{len(self.field_names)} fields")
    
    def get_pending_fields(self) -> List[str]:
        """Get list of fields not yet published"""
        return [f for f in self.field_names if f not in self.published_fields]
    
    def should_invoke_next(self) -> bool:
        """
        Check if we should invoke the next function now.
        
        Returns True after the first field is published and
        before we've invoked the next function.
        """
        with self._lock:
            return len(self.published_fields) == 1 and not self.next_function_invoked
    
    def mark_next_invoked(self):
        """Mark that the next function has been invoked"""
        with self._lock:
            self.next_function_invoked = True
            debug_log("Marked next function as invoked")
    
    def get_streaming_payload(self) -> dict:
        """
        Get payload with published values and futures for pending fields.
        
        Returns a dict that can be passed to the next function.
        """
        pending = self.get_pending_fields()
        return create_streaming_payload(
            self.session_id,
            self.source_function,
            self.published_fields.copy(),
            pending
        )
    
    def is_complete(self) -> bool:
        """Check if all fields have been published"""
        return len(self.published_fields) == len(self.field_names)


class LazyFutureDict(dict):
    """
    A dict that lazily resolves future values on access.
    
    This allows the receiver function to access fields normally,
    and futures are resolved only when actually accessed.
    """
    
    def __init__(self, data: dict):
        super().__init__()
        self._raw_data = data
        self._resolved = {}
        
    def __getitem__(self, key):
        if key in self._resolved:
            return self._resolved[key]
        
        if key in self._raw_data:
            value = self._raw_data[key]
            if is_future(value):
                debug_log(f"Lazily resolving future for key '{key}'")
                resolved = resolve_future(value)
                self._resolved[key] = resolved
                return resolved
            else:
                self._resolved[key] = value
                return value
        
        raise KeyError(key)
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    
    def __contains__(self, key):
        return key in self._raw_data
    
    def keys(self):
        return self._raw_data.keys()
    
    def values(self):
        return [self[k] for k in self.keys()]
    
    def items(self):
        return [(k, self[k]) for k in self.keys()]
    
    def __iter__(self):
        return iter(self._raw_data)
    
    def __len__(self):
        return len(self._raw_data)


def wrap_input_with_lazy_resolution(event: dict) -> dict:
    """
    Wrap event's input data with lazy future resolution.
    
    If the input contains futures, they will be resolved
    only when accessed by the handler.
    """
    if 'Data' not in event or 'Value' not in event.get('Data', {}):
        return event
    
    input_data = event['Data']['Value']
    
    # Check if any field is a future
    has_futures = any(is_future(v) for v in input_data.values() if isinstance(input_data, dict))
    
    if has_futures:
        debug_log(f"Input has futures, wrapping with lazy resolution")
        event['Data']['Value'] = LazyFutureDict(input_data)
    
    
    return event


def resolve_input_futures(event: dict) -> dict:
    """
    Eagerly resolve all futures in event's input data.
    
    Alternative to lazy resolution - resolves everything upfront.
    """
    if 'Data' not in event or 'Value' not in event.get('Data', {}):
        return event
    
    input_data = event['Data']['Value']
    
    if isinstance(input_data, dict):
        resolved_data = {}
        for key, value in input_data.items():
            if is_future(value):
                debug_log(f"Eagerly resolving future for key '{key}'")
                resolved_data[key] = resolve_future(value)
            else:
                resolved_data[key] = value
        event['Data']['Value'] = resolved_data
    
    return event


# =============================================================================
# GLOBAL STATE FOR STREAMING COORDINATION
# =============================================================================

# Thread-local storage for streaming output
_streaming_state = threading.local()

def set_streaming_output(payload: dict) -> None:
    """
    Set the streaming output payload AND invoke continuation immediately.
    
    Called by transformed user code after first field is computed.
    This will immediately invoke the next function with partial data.
    """
    _streaming_state.output = payload
    _streaming_state.invoked = False
    debug_log(f"Streaming output set with {len(payload)} fields")
    
    # Immediately invoke continuation if we have unum context
    unum_instance = getattr(_streaming_state, 'unum', None)
    event = getattr(_streaming_state, 'event', None)
    
    if unum_instance is not None and event is not None:
        invoke_streaming_continuation(unum_instance, event, payload)
    else:
        debug_log("No unum context available - continuation will be invoked by runtime")


def get_streaming_output() -> Optional[dict]:
    """
    Get the streaming output payload if set.
    
    Returns None if no streaming output was set.
    """
    return getattr(_streaming_state, 'output', None)


def clear_streaming_output() -> None:
    """
    Clear the streaming output state.
    """
    _streaming_state.output = None
    _streaming_state.invoked = False


def was_streaming_invoked() -> bool:
    """
    Check if streaming continuation was already invoked.
    """
    return getattr(_streaming_state, 'invoked', False)


def mark_streaming_invoked() -> None:
    """
    Mark that streaming continuation has been invoked.
    """
    _streaming_state.invoked = True


def register_unum_context(unum_instance, event: dict) -> None:
    """
    Register unum instance and event for streaming invocation.
    
    Called by main.py before invoking user function so that
    transformed code can trigger streaming continuation.
    """
    _streaming_state.unum = unum_instance
    _streaming_state.event = event
    _streaming_state.output = None
    _streaming_state.invoked = False
    debug_log(f"Unum context registered for streaming")


def unregister_unum_context() -> None:
    """
    Unregister unum context after user function completes.
    """
    _streaming_state.unum = None
    _streaming_state.event = None
    debug_log("Unum context unregistered")


def invoke_streaming_continuation(unum_instance, event: dict, streaming_payload: dict) -> bool:
    """
    Invoke continuation with streaming payload.
    
    This is called by the runtime (or transformed code) to send
    partial output to the next function.
    
    Args:
        unum_instance: The Unum runtime instance
        event: The original input event
        streaming_payload: The payload with values + future refs
        
    Returns:
        True if invocation was successful
    """
    if was_streaming_invoked():
        debug_log("Streaming continuation already invoked, skipping")
        return False
    
    try:
        session = unum_instance.get_session(event)
        next_payload_metadata = unum_instance.run_next_payload_modifiers(event)
        
        # Add streaming marker
        streaming_payload['__streaming__'] = True
        streaming_payload['Session'] = session
        
        gc_info = {
            unum_instance.get_my_instance_name(event): 
            unum_instance.get_my_outgoing_edges(event, streaming_payload)
        }
        
        # Invoke all continuations
        for c in unum_instance.cont_list:
            c.run(
                streaming_payload,
                session,
                next_payload_metadata,
                event,
                unum_instance.get_my_unum_index_list(event),
                gc=gc_info,
                my_name=unum_instance.name,
                my_curr_instance_name=unum_instance.get_my_instance_name(event)
            )
        
        mark_streaming_invoked()
        debug_log("Streaming continuation invoked successfully")
        return True
        
    except Exception as e:
        print(f"[STREAMING] Error invoking continuation: {e}")
        import traceback
        traceback.print_exc()
        return False
