import uuid
import time, datetime, json, os, math
import asyncio
from typing import Any, List, Dict, Optional, Tuple


def safe_asyncio_run(coro):
    """Safely run async code in Lambda environment (handles warm containers).
    
    AWS Lambda warm containers can have issues with asyncio.run() because
    it tries to close the event loop after each call. On subsequent invocations
    in the same container, there may be no current event loop in the thread.
    
    This function creates a new event loop, runs the coroutine, and properly
    cleans up, making it safe for repeated use in Lambda warm containers.
    """
    try:
        # Try to get existing loop (might work in some cases)
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        # If we have a running loop, we can't use run() - use run_until_complete
        if loop.is_running():
            # This shouldn't happen in Lambda, but handle it anyway
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop or loop is closed - create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


if os.environ['FAAS_PLATFORM'] == 'aws':
    import boto3
    from botocore.exceptions import ClientError
elif os.environ['FAAS_PLATFORM'] =='gcloud':
    from google.cloud import firestore
    from google.cloud import exceptions as gcloudexceptions


# =============================================================================
# LAZY INPUT (FUTURE/PROMISE PATTERN) FOR EAGER FAN-IN
# =============================================================================

class LazyInput:
    '''A Future/Promise-like wrapper for a single fan-in input.
    
    When the fan-in function is invoked eagerly (before all inputs are ready),
    each input is wrapped in a LazyInput. The actual data is fetched only when
    the user code tries to access it via .get() or .value property.
    
    If the data is not yet available, .get() will poll the datastore until
    the data becomes available or timeout is reached.
    
    Usage in user code:
        def lambda_handler(inputs, context):
            # inputs is a LazyInputList
            
            # Do initialization that doesn't need any inputs
            result = initialize_something()
            
            # Access first input - blocks if not ready
            data0 = inputs[0].get()  # or inputs[0].value
            
            # Process data0...
            intermediate = process(data0)
            
            # Access second input - blocks if not ready  
            data1 = inputs[1].get()
            
            # Continue processing...
            return combine(intermediate, data1)
    '''
    
    def __init__(self, datastore, session, instance_name, poll_interval=0.1, timeout=300, debug=False):
        '''
        @param datastore The datastore driver instance
        @param session The session ID
        @param instance_name The instance name of the branch whose output we need
        @param poll_interval Seconds between polls if data not ready
        @param timeout Maximum seconds to wait
        @param debug Whether to print debug messages
        '''
        self._datastore = datastore
        self._session = session
        self._instance_name = instance_name
        self._poll_interval = poll_interval
        self._timeout = timeout
        self._debug = debug
        
        # Cached values
        self._resolved = False
        self._data = None
        self._gc_info = None
    
    @property
    def is_resolved(self):
        '''Check if this input has been fetched (without blocking)'''
        return self._resolved
    
    @property
    def instance_name(self):
        '''Get the instance name of the branch'''
        return self._instance_name
    
    def try_resolve(self):
        '''Try to fetch the data without blocking.
        
        @return True if data was fetched, False if not available yet
        '''
        if self._resolved:
            return True
        
        # Try to read the checkpoint (use get_checkpoint_full for full data)
        checkpoint = self._datastore.get_checkpoint_full(self._session, self._instance_name)
        
        if checkpoint is not None:
            # Data is available - cache it
            self._data = checkpoint.get('User')
            self._gc_info = checkpoint.get('GC')
            self._resolved = True
            
            if self._debug:
                print(f'[DEBUG] LazyInput resolved: {self._instance_name}')
            
            return True
        
        return False
    
    def get(self, poll_interval=None, timeout=None):
        '''Get the data, waiting if necessary.
        
        If the data is not yet available, this will poll the datastore
        until it becomes available or timeout is reached.
        
        @param poll_interval Override default poll interval
        @param timeout Override default timeout
        @return The user data from this branch
        @raises TimeoutError if timeout exceeded
        '''
        if self._resolved:
            return self._data
        
        interval = poll_interval if poll_interval is not None else self._poll_interval
        max_wait = timeout if timeout is not None else self._timeout
        
        start_time = time.time()
        
        while True:
            if self.try_resolve():
                return self._data
            
            elapsed = time.time() - start_time
            if elapsed >= max_wait:
                raise TimeoutError(
                    f'Timeout waiting for input from {self._instance_name} '
                    f'after {elapsed:.2f}s'
                )
            
            if self._debug:
                print(f'[DEBUG] LazyInput waiting for {self._instance_name} ({elapsed:.1f}s elapsed)')
            
            time.sleep(interval)
    
    @property
    def value(self):
        '''Alias for get() - property access that waits for data'''
        return self.get()
    
    @property
    def gc_info(self):
        '''Get GC info (only available after resolution)'''
        if not self._resolved:
            self.get()  # Force resolution
        return self._gc_info
    
    def __repr__(self):
        status = "resolved" if self._resolved else "pending"
        return f'<LazyInput({self._instance_name}, {status})>'


class LazyInputList:
    '''A TRANSPARENT list-like container for eager fan-in inputs.
    
    This class behaves EXACTLY like a regular Python list from the user's
    perspective. When you access inputs[0], it automatically waits for the
    data if not ready and returns the actual value - NOT a wrapper object.
    
    This means fan-in functions don't need to be written differently!
    The eager fan-in optimization is completely transparent to user code.
    
    Example - user code is IDENTICAL for regular and eager fan-in:
        def lambda_handler(inputs, context):
            # Works exactly the same whether eager or not
            user_mentions = inputs[0]    # Blocks if not ready, returns data
            shortened_urls = inputs[1]   # Blocks if not ready, returns data
            
            # Iteration works normally too
            for data in inputs:
                process(data)
            
            return combine(user_mentions, shortened_urls)
    
    The only difference is WHEN the blocking happens:
    - Regular fan-in: All inputs fetched BEFORE lambda_handler is called
    - Eager fan-in: Each input fetched when first accessed (on-demand)
    '''
    
    def __init__(self, lazy_inputs, debug=False):
        '''
        @param lazy_inputs List of LazyInput objects
        @param debug Whether to print debug messages
        '''
        self._inputs = lazy_inputs
        self._debug = debug
    
    def __len__(self):
        return len(self._inputs)
    
    def __getitem__(self, index):
        '''Get data by index - AUTOMATICALLY waits if not ready.
        
        Returns the actual user data, not the LazyInput wrapper.
        This makes the list behave exactly like a regular Python list.
        '''
        if isinstance(index, slice):
            # Handle slicing
            return [inp.get() for inp in self._inputs[index]]
        return self._inputs[index].get()
    
    def __iter__(self):
        '''Iterate over the actual data values (waits for each as needed)'''
        for inp in self._inputs:
            yield inp.get()
    
    def __contains__(self, item):
        '''Check if item is in the list (resolves all inputs)'''
        for inp in self._inputs:
            if inp.get() == item:
                return True
        return False
    
    def index(self, value, start=0, stop=None):
        '''Find index of value (resolves inputs as needed)'''
        if stop is None:
            stop = len(self._inputs)
        for i in range(start, stop):
            if self._inputs[i].get() == value:
                return i
        raise ValueError(f'{value} is not in list')
    
    def count(self, value):
        '''Count occurrences of value (resolves all inputs)'''
        return sum(1 for inp in self._inputs if inp.get() == value)
    
    # ---- Methods for advanced usage (optional, not needed for normal code) ----
    
    def get_all(self, poll_interval=None, timeout=None):
        '''Get all values as a regular list, waiting for any that aren't ready.
        
        @return List of user data values in order
        '''
        return [inp.get(poll_interval, timeout) for inp in self._inputs]
    
    def to_list(self):
        '''Convert to a regular Python list (waits for all inputs)'''
        return self.get_all()
    
    def is_ready(self, index):
        '''Check if a specific input is ready WITHOUT blocking'''
        return self._inputs[index].is_resolved or self._inputs[index].try_resolve()
    
    def all_ready(self):
        '''Check if ALL inputs are ready WITHOUT blocking'''
        for inp in self._inputs:
            if not inp.is_resolved and not inp.try_resolve():
                return False
        return True
    
    def get_resolved(self):
        '''Get values that are already resolved without blocking.
        
        @return Dict mapping index to value for resolved inputs
        '''
        result = {}
        for i, inp in enumerate(self._inputs):
            if inp.is_resolved:
                result[i] = inp._data
        return result
    
    def get_pending_indices(self):
        '''Get indices of inputs that aren't resolved yet.
        
        @return List of indices
        '''
        return [i for i, inp in enumerate(self._inputs) if not inp.is_resolved]
    
    def try_resolve_all(self):
        '''Try to resolve all inputs without blocking.
        
        @return Tuple of (num_resolved, num_pending)
        '''
        resolved = 0
        pending = 0
        for inp in self._inputs:
            if inp.try_resolve():
                resolved += 1
            else:
                pending += 1
        return resolved, pending
    
    def get_gc_tasks(self):
        '''Get GC info from all resolved inputs.
        
        @return Dict of GC tasks (waits for all inputs to resolve)
        '''
        gc_tasks = {}
        for inp in self._inputs:
            gc_info = inp.gc_info  # This will wait if not resolved
            if gc_info:
                gc_tasks.update(gc_info)
        return gc_tasks
    
    def __repr__(self):
        resolved = sum(1 for inp in self._inputs if inp.is_resolved)
        return f'<LazyInputList({resolved}/{len(self._inputs)} resolved)>'


# =============================================================================
# FUTURE-BASED EXECUTION (ASYNC PATTERN) FOR EAGER FAN-IN
# =============================================================================
# This implements the true Future-Based execution pattern using asyncio.Event()
# for non-blocking waiting. The key difference from LazyInput:
#   - LazyInput: Synchronous polling (blocks the thread)
#   - UnumFuture: Async waiting (yields control to event loop)
# =============================================================================

class UnumFuture:
    """A true Future/Promise implementation using asyncio.Event().
    
    This is the core primitive for Future-Based execution. Each UnumFuture
    wraps either:
    - A ready value (immediately available)
    - A pending value (will arrive when the branch finishes)
    """
    
    def __init__(
        self,
        datastore=None,
        session: str = None,
        instance_name: str = None,
        value: Any = None,
        is_ready: bool = False,
        poll_interval: float = 0.1,
        timeout: float = 300,
        debug: bool = False
    ):
        self._datastore = datastore
        self._session = session
        self._instance_name = instance_name
        self._value = value
        self._is_ready = is_ready
        self._poll_interval = poll_interval
        self._timeout = timeout
        self._debug = debug
        self._gc_info = None
        
        # Python 3.10+ doesn't require an event loop to create asyncio.Event()
        self._event = asyncio.Event()
        if is_ready:
            self._event.set()
    
    @property
    def is_ready(self) -> bool:
        return self._is_ready
    
    @property
    def instance_name(self) -> str:
        return self._instance_name
    
    @property
    def gc_info(self) -> Optional[Dict]:
        return self._gc_info
    
    def set_value(self, value: Any, gc_info: Dict = None):
        self._value = value
        self._gc_info = gc_info
        self._is_ready = True
        self._event.set()
        
        if self._debug:
            print(f'[DEBUG] UnumFuture resolved: {self._instance_name}')
    
    def get_value_sync(self) -> Any:
        if not self._is_ready:
            raise ValueError(f'Value not ready for {self._instance_name}. Use await_value() for async waiting.')
        return self._value
    
    async def await_value(self) -> Any:
        if self._is_ready:
            return self._value
        
        start_time = time.time()
        
        while not self._is_ready:
            if await self._poll_datastore():
                break
            
            elapsed = time.time() - start_time
            if elapsed >= self._timeout:
                raise TimeoutError(
                    f'Timeout waiting for {self._instance_name} after {elapsed:.2f}s'
                )
            
            if self._debug:
                print(f'[DEBUG] UnumFuture waiting for {self._instance_name} ({elapsed:.1f}s elapsed)')
            
            await asyncio.sleep(self._poll_interval)
        
        return self._value
    
    async def _poll_datastore(self) -> bool:
        if self._datastore is None:
            return False
        
        checkpoint = self._datastore.get_checkpoint_full(self._session, self._instance_name)
        
        if checkpoint is not None:
            self.set_value(
                value=checkpoint.get('User'),
                gc_info=checkpoint.get('GC')
            )
            return True
        
        return False
    
    def try_resolve(self) -> bool:
        if self._is_ready:
            return True
        
        if self._datastore is None:
            return False
        
        checkpoint = self._datastore.get_checkpoint_full(self._session, self._instance_name)
        
        if checkpoint is not None:
            self.set_value(
                value=checkpoint.get('User'),
                gc_info=checkpoint.get('GC')
            )
            return True
        
        return False
    
    def __repr__(self):
        status = "ready" if self._is_ready else "pending"
        return f'<UnumFuture({self._instance_name}, {status})>'


class AsyncFutureInputList:
    """A list-like container for Future-Based fan-in inputs (async pattern).
    
    This class supports both sync and async access patterns.
    
    OPTIMIZATION: We resolve ALL pending futures in a SINGLE asyncio.run() call
    instead of calling asyncio.run() for each input access. This avoids the
    expensive overhead of creating/destroying event loops multiple times.
    """
    
    def __init__(self, futures: List[UnumFuture], debug: bool = False):
        self._futures = futures
        self._debug = debug
        self._resolved_values = {}  # Cache for resolved values
        self._all_resolved = False
    
    def __len__(self) -> int:
        return len(self._futures)
    
    def _resolve_all_sync(self):
        """Resolve ALL pending futures in a single asyncio.run() call.
        
        This is much more efficient than calling asyncio.run() per input.
        """
        if self._all_resolved:
            return
        
        # First, cache any already-resolved values
        for i, future in enumerate(self._futures):
            if future.is_ready and i not in self._resolved_values:
                self._resolved_values[i] = future.get_value_sync()
        
        # Find pending futures
        pending_indices = [i for i, f in enumerate(self._futures) if i not in self._resolved_values]
        
        if not pending_indices:
            self._all_resolved = True
            return
        
        # Resolve ALL pending futures in a SINGLE event loop (Lambda-safe)
        async def resolve_all_pending():
            tasks = [self._futures[i].await_value() for i in pending_indices]
            return await asyncio.gather(*tasks)
        
        try:
            results = safe_asyncio_run(resolve_all_pending())
            for i, idx in enumerate(pending_indices):
                self._resolved_values[idx] = results[i]
        except Exception as e:
            if self._debug:
                print(f'[DEBUG] Error resolving futures: {e}')
            raise
        
        self._all_resolved = True
    
    # ASYNC ACCESS
    async def get_async(self, index: int) -> Any:
        return await self._futures[index].await_value()
    
    async def get_all_async(self) -> List[Any]:
        return await asyncio.gather(*[f.await_value() for f in self._futures])
    
    async def __aiter__(self):
        for future in self._futures:
            yield await future.await_value()
    
    # SYNC ACCESS - PARALLEL BACKGROUND POLLING
    # When waiting for index X, also poll ALL other pending futures in parallel.
    # This way, by the time user code processes inputs[0] and asks for inputs[1],
    # inputs[1] might already be resolved from background polling.
    
    def _resolve_with_background_polling(self, target_index: int):
        """Resolve target_index while polling ALL pending futures in parallel.
        
        This is the FASTEST approach:
        - Wait specifically for target_index
        - But also poll all other pending futures in the background
        - When target is ready, return immediately (background tasks continue)
        - Future accesses benefit from background resolution
        """
        if target_index in self._resolved_values:
            return
        
        if self._futures[target_index].is_ready:
            self._resolved_values[target_index] = self._futures[target_index].get_value_sync()
            return
        
        # Find ALL pending indices (including target)
        pending_indices = [
            i for i in range(len(self._futures))
            if i not in self._resolved_values and not self._futures[i].is_ready
        ]
        
        if not pending_indices:
            # Target must be ready now
            self._resolved_values[target_index] = self._futures[target_index].get_value_sync()
            return
        
        async def resolve_target_with_background():
            """Poll all pending futures, return when target is ready."""
            
            # Create tasks for ALL pending futures
            tasks = {
                i: asyncio.create_task(self._futures[i].await_value())
                for i in pending_indices
            }
            
            # Wait until our target is done, but let others run in background
            target_task = tasks[target_index]
            
            # Use asyncio.wait to wait for target while others poll in parallel
            done, pending_tasks = await asyncio.wait(
                tasks.values(),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Collect results from completed tasks
            results = {}
            for i, task in tasks.items():
                if task.done() and not task.cancelled():
                    try:
                        results[i] = task.result()
                    except Exception:
                        pass
            
            # If target not done yet, keep waiting for it specifically
            # while other tasks continue in background
            while not target_task.done():
                done, pending_tasks = await asyncio.wait(
                    [t for t in tasks.values() if not t.done()],
                    return_when=asyncio.FIRST_COMPLETED
                )
                # Collect newly completed results
                for i, task in tasks.items():
                    if i not in results and task.done() and not task.cancelled():
                        try:
                            results[i] = task.result()
                        except Exception:
                            pass
            
            # Final collection of all completed tasks
            for i, task in tasks.items():
                if i not in results and task.done() and not task.cancelled():
                    try:
                        results[i] = task.result()
                    except Exception:
                        pass
            
            return results
        
        try:
            results = safe_asyncio_run(resolve_target_with_background())
            # Cache ALL resolved values (not just target)
            for i, value in results.items():
                self._resolved_values[i] = value
        except Exception as e:
            if self._debug:
                print(f'[DEBUG] Error resolving future[{target_index}]: {e}')
            raise
    
    def __getitem__(self, index) -> Any:
        if isinstance(index, slice):
            # For slices, resolve all indices in parallel
            indices = list(range(*index.indices(len(self._futures))))
            if indices:
                # Use parallel resolution for slice
                self._resolve_with_background_polling(indices[0])
            results = []
            for i in indices:
                if i not in self._resolved_values:
                    self._resolve_with_background_polling(i)
                results.append(self._resolved_values[i])
            return results
        
        # Check if already resolved (cached from background polling)
        if index in self._resolved_values:
            return self._resolved_values[index]
        
        # Check if the future is ready (no async needed)
        if self._futures[index].is_ready:
            self._resolved_values[index] = self._futures[index].get_value_sync()
            return self._resolved_values[index]
        
        # PARALLEL BACKGROUND POLLING: Wait for this index,
        # but poll ALL pending futures in parallel so future accesses are faster
        self._resolve_with_background_polling(index)
        return self._resolved_values[index]
    
    def __iter__(self):
        """Iterate over values - polls ALL in background for speed."""
        # On first iteration, start polling all in background
        # Each yield returns as soon as that index is ready
        for i in range(len(self._futures)):
            if i not in self._resolved_values:
                self._resolve_with_background_polling(i)
            yield self._resolved_values[i]
    
    # Utility methods
    def is_ready(self, index: int) -> bool:
        return self._futures[index].is_ready or index in self._resolved_values
    
    def all_ready(self) -> bool:
        return all(f.is_ready for f in self._futures)
    
    def get_ready_count(self) -> Tuple[int, int]:
        ready = sum(1 for f in self._futures if f.is_ready)
        return ready, len(self._futures) - ready
    
    def get_all(self) -> List[Any]:
        """Get ALL values, resolving any pending in PARALLEL.
        
        Use this when you know you need all inputs - it's more efficient
        to resolve all pending futures in parallel than one-by-one.
        
        Example:
            # If you need all inputs anyway, this is faster:
            all_data = inputs.get_all()
            
            # Than accessing each individually:
            data0 = inputs[0]  # waits for 0
            data1 = inputs[1]  # waits for 1
            data2 = inputs[2]  # waits for 2
        """
        self._resolve_all_sync()
        return [self._resolved_values[i] for i in range(len(self._futures))]
    
    def try_resolve_all(self) -> Tuple[int, int]:
        resolved = 0
        pending = 0
        for i, future in enumerate(self._futures):
            if future.try_resolve():
                self._resolved_values[i] = future.get_value_sync()
                resolved += 1
            else:
                pending += 1
        return resolved, pending
    
    def get_gc_tasks(self) -> Dict:
        self._resolve_all_sync()
        gc_tasks = {}
        for future in self._futures:
            if future.gc_info:
                gc_tasks.update(future.gc_info)
        return gc_tasks
    
    def get_futures(self) -> List[UnumFuture]:
        return self._futures
    
    def __repr__(self):
        ready, pending = self.get_ready_count()
        return f'<AsyncFutureInputList({ready}/{len(self._futures)} ready)>'


class UnumIntermediaryDataStore(object):
    
    subclasses = {}

    def __init__(self, ds_type, ds_name, debug):
        '''
            @param type s3|dynamodb|redis|elasticache|fs|efs
            @param name s3 bucket | dynamodb table
        '''
        self.my_type = ds_type
        self.name = ds_name
        self.debug = debug


    @classmethod
    def add_datastore(cls, datastore_type):
        def wrapper(subclass):
            cls.subclasses[datastore_type] = subclass
            return subclass

        return wrapper


    @classmethod
    def create(cls, datastore_type, *params):
        if datastore_type not in cls.subclasses:
            raise ValueError(f'unum does not support {platform} as intermediary data store')

        return cls.subclasses[datastore_type](*params)



@UnumIntermediaryDataStore.add_datastore('firestore')
class FirestoreDriver(UnumIntermediaryDataStore):
    '''
    In the gcloud Firestore implementation, each session is saved in its own
    collection. The collection name is the session id. Checkpoints are
    documents in the collection with the function's instance name as its
    document name.
    '''
    def __init__(self, ds_name, debug):
        super(FirestoreDriver, self).__init__("firestore", ds_name, debug)
        self.db = firestore.Client()



    def _read(self, collection, document):
        '''Read a single document from a collection
        '''
        doc_ref = self.db.collection(u'{}'.format(collection)).document(u'{}'.format(document))

        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            return None



    def read_input(self, collection, documents):
        '''Read multiple documents from a collection

        Used by the aggregation function to read its inputs
        '''
        if self.debug:
            print(f'[DEBUG] Reading inputs from collection: {collection}, and documents: {documents}')
        
        return [self._read(collection, d) for d in documents]



    def get_checkpoint(self, session, instance_name):
        return self._read(session, instance_name)



    def _create_if_not_exist(self, collection, document, value):
        '''
        According to
        https://googleapis.dev/python/firestore/latest/document.html#google.cloud.firestore_v1.document.DocumentReference.create,
        create() will fail with google.cloud.exceptions.Conflict if the
        document already exists.

        It's not fully clear whether create is strongly consistent in that if
        I have 2 concurrent threads calling create, does it guarantee that one
        of the create() calls will fail with google.cloud.exceptions.Conflict?
        '''

        doc_ref = self.db.collection(u'{}'.format(collection)).document(u'{}'.format(document))

        try:
            doc_ref.create(value)
            return 1
        except gcloudexceptions.Conflict as e:
            return -1
        except Exception as e:
            print(f'[ERROR] Checkpointing encountered unexpected error: {e}')
            return -2


    def checkpoint(self, session, instance_name, data):
        '''

        @return 1 if successful. -1 if a checkpoint already exists. -2 if
            other errors.
        '''

        return self._create_if_not_exist(session, instance_name, data)



    def _delete(self, collection, document):
        '''Delete a document in a collection
        '''
        doc_ref = self.db.collection(u'{}'.format(collection)).document(u'{}'.format(document))

        try:
            doc_ref.delete()
            return 1

        except ClientError as e:
            raise e



    def delete_checkpoint(self, session, instance_name):
        return self._delete(session, instance_name)



    def gc_sync_point_name(self, session, parent_function_instance_name):
        return session, f'{parent_function_instance_name}-gc'



    def fanin_sync_point_name(self, session, aggregation_function_instance_name):
        return session, f'{aggregation_function_instance_name}-fanin'



    def gc_sync_ready(self, session, parent_function_instance_name, index, my_instance_name, num_branches):
        '''Mark my gc as ready and check if gc is ready to run

        In the case of a parent node invoking multiple downstream child nodes,
        all child nodes need to have created their checkpoints before the
        parent node's checkpoint can be garbage collected. Unum have all child
        nodes synchronize via the intermediate datastore so that the
        last-to-finish child node deletes the parent's checkpoint.

        The synchronization item is named after the parent function's instance
        name. In practice, in Firestore, the collection name is the session id
        and the document name is the parent function's instance name with a
        "-gc" suffix.

        @return True is I'm the last-to-finish child node. False if I'm not.
        '''
        return self._sync_ready(self.gc_sync_point_name(session, parent_function_instance_name), index, num_branches)



    def fanin_sync_ready(self, session, aggregation_function_instance_name, index, my_instance_name, num_branches):
        '''Mark my branch as ready and check if fan-in is ready to run

        In the case of fan-in, all upstream branches need to have created
        their checkpoints before the aggregation function is invoked. Unum
        have all branches synchronize via the intermediate datastore so that only
        the last-to-finish branch invokes the aggregation function.

        The synchronization item is named after the aggregation function's instance
        name. In practice, in Firestore, the collection name is the session id
        and the document name is the aggregation function's instance name with a
        "-fanin" suffix.

        @return True is I'm the last-to-finish branch. False if I'm not.
        '''
        return self._sync_ready(self.fanin_sync_point_name(session, aggregation_function_instance_name), index, my_instance_name, num_branches)



    def _sync_ready(self, sync_point_name, index, my_instance_name, num_branches):
        '''Mark the caller ready and return whether all branches are ready.

        @param sync_point_name tuple of session ID (as the Firestore
            collection name) and the synchronization object name (as the
            Firestore document name)
        @param index caller's index in the synchronization object
        @param num_branches the number of nodes that need to synchronize

        @return True if all branches are ready. False if not.
        '''
        # return self._sync_ready_bitmap(sync_point_name, index, num_branches)
        return self._sync_ready_set(sync_point_name, my_instance_name, num_branches)



    def _sync_ready_set(self, sync_point_name, my_instance_name, num_branches):
        '''Synchronize using a set in Firestore

        Based on
        https://cloud.google.com/firestore/docs/manage-data/add-data#update_elements_in_an_array,
        we can implement a Set using a Firestore array with the `arrayUnion()`
        API.
        '''
        self._create_set(sync_point_name)
        ready_set = self._update_set_result(sync_point_name, my_instance_name)

        return self._set_ready(ready_set, num_branches)



    def _create_set(self, set_name):
        '''Create a document with an array field named "ReadySet" initialized
        to an empty array

        @param set_name tuple containing the session ID and the document
            name of the set to be created
        '''
        collection = set_name[0]
        document = set_name[1]

        if self.debug:
            print(f'[DEBUG] Creating ready set. Collection: {collection}. Document: {document}')

        value = {'ReadySet':[]}

        return self._create_if_not_exist(collection, document, value)



    def _update_set_result(self, set_name, my_instance_name):
        '''Add my instance name to the ready set
        '''
        collection = set_name[0]
        document = set_name[1]

        if self.debug:
            print(f'[DEBUG] Adding {my_instance_name} to ready set ({collection}, {document})')

        set_ref = self.db.collection(collection).document(document)

        result = set_ref.update({'ReadySet': firestore.ArrayUnion([my_instance_name])})

        if self.debug:
            print(f'[DEBUG] Set result after update: {type(result)}')
            print(f'{type(result.transform_results)}')
            for e in result.transform_results:
                print(e)
                print(type(e))

        return result



    def _set_ready(self, ready_set, target_size):
        return len(ready_set) == target_size



    def _sync_ready_bitmap(self, sync_point_name, index, num_branches):
        '''Synchronize using a bitmap in Firestore

        First create a bitmap if it does not yet exist.

        Second flip the bit at index `index` in the bitmap.

        Finally return if the bitmap is all 1's after the write in the 2nd
        step

        @param sync_point_name tuple of session ID and the document name of
            the bitmap object to be created
        @param index int the index of this function in the bitmap.
            self._update_bitmap_result will flip the bit at this index
        @param num_branches int total number branches which is the length of
            the bitmap to be created

        @return True if all branches are ready. False if not.
        '''
        self._create_bitmap(sync_point_name, num_branches)
        ready_map = self._update_bitmap_result(sync_point_name, index)

        return self._bitmap_ready(ready_map)



    def _create_bitmap(self, bitmap_name, bitmap_size):
        '''Create a document with an array of booleans initialized to False

        @param bitmap_name tuple containing the session ID and the document
            name of the bitmap to be created
        @param bitmap_size int the length of the boolean array
        '''
        collection = bitmap_name[0]
        document = bitmap_name[1]

        if self.debug:
            print(f'creating collection: {collection} and document: {document} as bitmap of length {bitmap_size}')

        value = {"ReadyMap": [False for i in range(bitmap_size)]}

        return self._create_if_not_exist(collection, document, value)



    def _update_bitmap_result(self, bitmap_name, index):

        # The default max retry attempts for Firestore transaction is only 5.
        # It is too low for fan-outs and I start to see transactions fail
        # around 15 parallel branches.
        transaction = self.db.transaction(max_attempts=500)
        bitmap_ref = self.db.collection(bitmap_name[0]).document(bitmap_name[1])

        @firestore.transactional
        def _update_my_index(transaction, bitmap_ref):
            snapshot = bitmap_ref.get(transaction=transaction)
            ready_map = snapshot.get('ReadyMap')

            # print(f'ReadyMap in snapshot: {ready_map}')

            ready_map[index] = True
            transaction.update(bitmap_ref, {
                'ReadyMap': ready_map
            })


            return ready_map

        try:
            result = _update_my_index(transaction, bitmap_ref)
        except Exception as e:
            # If the prior transaction failed, retry after a second
            result = _update_my_index(transaction, bitmap_ref)
            return result
        else:
            return result



    def _bitmap_ready(self, bitmap):
        '''Check if the bitmap is all True
        '''
        for b in bitmap:
            if b == False:
                return False
        return True


    # =========================================================================
    # EAGER FAN-IN SUPPORT (Firestore)
    # These methods enable the eager fan-in pattern where the fan-in function
    # is invoked immediately by the first branch to complete, and then waits
    # for remaining inputs by polling the datastore.
    # =========================================================================

    def try_claim_eager_fanin(self, session, aggregation_function_instance_name):
        '''Atomically try to claim the right to invoke the fan-in function.
        
        Uses Firestore's create() which fails if document exists.
        
        @return True if this caller claimed the right to invoke, False otherwise
        '''
        claim_doc = f'{aggregation_function_instance_name}-eager-claim'
        
        try:
            result = self._create_if_not_exist(session, claim_doc, {
                'ClaimedAt': datetime.datetime.now().isoformat(),
                'Claimed': True
            })
            if result == 1:
                if self.debug:
                    print(f'[DEBUG] Successfully claimed eager fan-in for {aggregation_function_instance_name}')
                return True
            else:
                if self.debug:
                    print(f'[DEBUG] Eager fan-in already claimed for {aggregation_function_instance_name}')
                return False
        except Exception as e:
            if self.debug:
                print(f'[DEBUG] Error claiming eager fan-in: {e}')
            return False


    def check_checkpoints_exist(self, session, instance_names):
        '''Check which checkpoints exist in the datastore.
        
        @param session The session ID (collection name in Firestore)
        @param instance_names List of instance names to check
        @return Tuple of (ready_list, missing_list)
        '''
        ready = []
        missing = []
        
        for name in instance_names:
            doc = self._read(session, name)
            if doc is not None:
                ready.append(name)
            else:
                missing.append(name)
        
        return ready, missing


    def await_checkpoints(self, session, instance_names, poll_interval=0.1, timeout=300):
        '''Poll the datastore until all required checkpoints are available.
        
        @param session The session ID
        @param instance_names List of instance names whose checkpoints are needed
        @param poll_interval Seconds between polls
        @param timeout Maximum seconds to wait
        @return True when all checkpoints are available
        @raises TimeoutError if timeout is exceeded
        '''
        start_time = time.time()
        
        while True:
            ready, missing = self.check_checkpoints_exist(session, instance_names)
            
            if not missing:
                if self.debug:
                    print(f'[DEBUG] All {len(instance_names)} checkpoints ready after {time.time() - start_time:.2f}s')
                return True
            
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f'Timeout waiting for checkpoints after {elapsed:.2f}s. '
                    f'Missing: {missing}'
                )
            
            if self.debug:
                print(f'[DEBUG] Waiting for {len(missing)} checkpoints: {missing[:3]}{"..." if len(missing) > 3 else ""}')
            
            time.sleep(poll_interval)


    def read_input_with_await(self, session, values, poll_interval=0.1, timeout=300):
        '''Read inputs for fan-in, waiting for any that aren't ready yet.
        '''
        self.await_checkpoints(session, values, poll_interval, timeout)
        return self.read_input(session, values)


    def get_checkpoint_full(self, session, instance_name):
        '''Get the full checkpoint document including User data and GC info.
        
        Used by LazyInput to fetch a single input's data on-demand.
        '''
        doc = self._read(session, instance_name)
        if doc is None:
            return None
        
        result = {'User': doc.get('User', doc)}
        if 'GC' in doc:
            result['GC'] = doc['GC']
        
        return result


    def create_lazy_inputs(self, session, instance_names, poll_interval=0.1, timeout=300):
        '''Create a LazyInputList for eager fan-in (Firestore version).
        '''
        lazy_inputs = [
            LazyInput(
                datastore=self,
                session=session,
                instance_name=name,
                poll_interval=poll_interval,
                timeout=timeout,
                debug=self.debug
            )
            for name in instance_names
        ]
        
        return LazyInputList(lazy_inputs, debug=self.debug)


    def create_future_inputs(self, session, instance_names, ready_names=None, poll_interval=0.1, timeout=300):
        '''Create an AsyncFutureInputList for Future-Based execution (Firestore version).
        
        @param session The session ID
        @param instance_names List of instance names for the fan-in inputs
        @param ready_names Optional list of names already known to be ready
        @param poll_interval Default poll interval for each UnumFuture
        @param timeout Default timeout for each UnumFuture
        @return AsyncFutureInputList containing UnumFuture objects
        '''
        ready_set = set(ready_names) if ready_names else set()
        
        futures = []
        for name in instance_names:
            is_ready = name in ready_set
            
            if is_ready:
                checkpoint = self.get_checkpoint_full(session, name)
                futures.append(UnumFuture(
                    datastore=self,
                    session=session,
                    instance_name=name,
                    value=checkpoint.get('User') if checkpoint else None,
                    is_ready=True,
                    poll_interval=poll_interval,
                    timeout=timeout,
                    debug=self.debug
                ))
            else:
                futures.append(UnumFuture(
                    datastore=self,
                    session=session,
                    instance_name=name,
                    is_ready=False,
                    poll_interval=poll_interval,
                    timeout=timeout,
                    debug=self.debug
                ))
        
        return AsyncFutureInputList(futures, debug=self.debug)


    def test(self):

        print(f'Firestore test')
        doc_ref = self.db.collection(u'users').document(u'alovelace')
        doc_ref.set({
            u'first': u'Ada',
            u'last': u'Lovelace',
            u'born': 1815
        })

        doc_ref = self.db.collection(u'users').document(u'aturing')
        doc_ref.set({
            u'first': u'Alan',
            u'middle': u'Mathison',
            u'last': u'Turing',
            u'born': 1912
        })

        users_ref = self.db.collection(u'users')
        docs = users_ref.stream()

        for doc in docs:
            print(f'{doc.id} => {doc.to_dict()}')

        session = f'{uuid.uuid4()}'

        self.checkpoint(session, 'test-function', {"output":"foo"})
        self.checkpoint(session, 'test-function', {"output":"foo"})




@UnumIntermediaryDataStore.add_datastore('dynamodb')
class DynamoDBDriver(UnumIntermediaryDataStore):


    def __init__(self, ds_name, debug):
        super(DynamoDBDriver, self).__init__("dynamodb", ds_name, debug)
        self.client = boto3.client('dynamodb')
        self.resource = boto3.resource('dynamodb')
        self.table = self.resource.Table(self.name)


    def read_input(self, session, values):
        '''Given the session id and a list of pointers to the intermediary
        data store, read all data and return them as an ordered list.

        Data in the returned list should correspond to the pointers in the
        `values` parameter *in the same order*.

        In practice, this function is only used by aggregation functions
        (fan-ins) to read its inputs.

        Elements in the `values` list are *instance names*.

        The pointers are used as is. It is the invoker's responsibility to
        expand the pointers and make sure that they are valid.
        Correspondingly, the IR of the fan-in branches, specifically the
        `Values` field that lists all fan-in branches, are written from the
        perspective of the branches (i.e., invokers).

        unum's fan-in semantics requires that the fan-in function be invoked
        only when ALL its inputs are available. Therefore, if one of the
        pointers in the `values` list doesn't exist in the data store, this
        function will throw an exception.

        On AWS, there's no reason to pass in a single data pointer when
        invoking a Lambda because asynchronous HTTP requests achieves the same
        results by adding the data onto Lambda's event queue. Therefore, the
        `values` parameter should always be a list. We don't consider the
        scenario where `values` is a dict.
        '''
        item_names = [f'{session}/{v}-output' for v in values]
        request_keys = [{'Name': k} for k in item_names]

        '''
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html#DynamoDB.ServiceResource.batch_get_item
        A example response of batch_get_item():

            {
                'Responses': {
                    'unum-dynamo-test-table': [
                        {
                            'Value': 'Hardcoded client put_item()',
                            'Name': '2c48bf10-fecf-4832-b25a-db1d4b9df840/Client-output'
                        },
                        {
                            'Value': 'Hardcoded Table put_item()',
                            'Name': '2c48bf10-fecf-4832-b25a-db1d4b9df840/Table-output'
                        }
                    ]
                },
                'UnprocessedKeys': {},
                'ResponseMetadata': {
                    'RequestId': 'UNE6VCORLFIED1LISTBG0M6VNJVV4KQNSO5AEMVJF66Q9ASUAAJG',
                    'HTTPStatusCode': 200,
                    'HTTPHeaders': {
                        'server': 'Server',
                        'date': 'Sun, 03 Oct 2021 02:15:41 GMT',
                        'content-type': 'application/x-amz-json-1.0',
                        'content-length': '285',
                        'connection': 'keep-alive',
                        'x-amzn-requestid': 'UNE6VCORLFIED1LISTBG0M6VNJVV4KQNSO5AEMVJF66Q9ASUAAJG',
                        'x-amz-crc32': '4084553720'
                    },
                    'RetryAttempts': 0
                }
            }

        NOTE: dynamodb.resource.batch_get_item() does NOT tell you which
        requested item it didn't find in the table.

        NOTE: If you request more than 100 items, BatchGetItem returns a
        ValidationException with the message "Too many items requested for the
        BatchGetItem call."

        NOTE: A single operation can retrieve up to 16 MB of data.
        BatchGetItem returns a partial result if the response size limit is
        exceeded. If a partial result is returned, the operation returns a
        value for UnprocessedKeys. You can use this value to retry the
        operation starting with the next item to get.

        NOTE: BatchGetItem returns a partial result if the table's provisioned
        throughput is exceeded, or an internal processing failure occurs. If a
        partial result is returned, the operation returns a value for
        UnprocessedKeys . You can use this value to retry the operation
        starting with the next item to get.

        If none of the items can be processed due to insufficient provisioned
        throughput on all of the tables in the request, then BatchGetItem
        returns a ProvisionedThroughputExceededException . If at least one of
        the items is successfully processed, then BatchGetItem completes
        successfully, while returning the keys of the unread items in
        UnprocessedKeys .
        '''

        all_ret = []

        for i in range(math.ceil(len(request_keys)/100)):
            this_batch = request_keys[i*100:(i+1)*100]

            this_batch_items = self.resource.batch_get_item(
                RequestItems={
                    self.name: {
                        'Keys': this_batch,
                        'ConsistentRead': True,
                    }
                })

            try:
                ret = this_batch_items['Responses'][self.name]
                all_ret = all_ret+ret

            except KeyError as e:
                print(this_batch_items)
                raise e

        # return a sorted array by the originally requested order
        order = {n: i for i, n in enumerate(item_names)}

        vals = []

        for e in sorted(all_ret, key=lambda d: order[d['Name']]):
            item = {
                'User': json.loads(e['User']),
                'Name': e['Name']
            }
            if 'GC' in e:
                item['GC'] = e['GC']

            vals.append(item)


        if len(vals) < len(values):
            print(f'[WARN] Not all values for fan-in were read from {self.my_type}')
            print(f'[WARN] Expect {len(values)}. Got {len(vals)}')
            print(all_ret)
        elif len(vals) > len(values):
            print(f'[WARN] More fan-in values read from {self.my_type} than expanded')

        return vals


    # =========================================================================
    # EAGER FAN-IN SUPPORT
    # These methods enable the eager fan-in pattern where the fan-in function
    # is invoked immediately by the first branch to complete, and then waits
    # for remaining inputs by polling the datastore.
    # =========================================================================

    def try_claim_eager_fanin(self, session, aggregation_function_instance_name):
        '''Atomically try to claim the right to invoke the fan-in function.
        
        The first branch to call this successfully "wins" and should invoke
        the fan-in function. All other branches will get False.
        
        Uses DynamoDB conditional write to ensure exactly-once semantics.
        
        @param session The session ID
        @param aggregation_function_instance_name The name of the fan-in function instance
        @return True if this caller claimed the right to invoke, False otherwise
        '''
        claim_name = f'{session}/{aggregation_function_instance_name}-eager-claim'
        
        try:
            self.table.put_item(
                Item={
                    "Name": claim_name,
                    "ClaimedAt": datetime.datetime.now().isoformat(),
                    "Claimed": True
                },
                ConditionExpression='attribute_not_exists(#N)',
                ExpressionAttributeNames={"#N": "Name"}
            )
            if self.debug:
                print(f'[DEBUG] Successfully claimed eager fan-in for {aggregation_function_instance_name}')
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                if self.debug:
                    print(f'[DEBUG] Eager fan-in already claimed for {aggregation_function_instance_name}')
                return False
            raise e


    def check_checkpoints_exist(self, session, instance_names):
        '''Check which checkpoints exist in the datastore.
        
        @param session The session ID
        @param instance_names List of instance names to check
        @return Tuple of (ready_list, missing_list) where ready_list contains
                names that exist and missing_list contains names that don't
        '''
        item_names = [f'{session}/{v}-output' for v in instance_names]
        request_keys = [{'Name': k} for k in item_names]
        
        ready = []
        missing = []
        
        for i in range(math.ceil(len(request_keys)/100)):
            this_batch = request_keys[i*100:(i+1)*100]
            this_batch_names = [k['Name'] for k in this_batch]
            
            try:
                response = self.resource.batch_get_item(
                    RequestItems={
                        self.name: {
                            'Keys': this_batch,
                            'ConsistentRead': True,
                            'ProjectionExpression': '#N',
                            'ExpressionAttributeNames': {'#N': 'Name'}
                        }
                    })
                
                found_names = {item['Name'] for item in response.get('Responses', {}).get(self.name, [])}
                
                for name in this_batch_names:
                    instance_name = name.replace(f'{session}/', '').replace('-output', '')
                    if name in found_names:
                        ready.append(instance_name)
                    else:
                        missing.append(instance_name)
                        
            except Exception as e:
                if self.debug:
                    print(f'[DEBUG] Error checking checkpoints: {e}')
                # On error, assume all in this batch are missing
                for name in this_batch_names:
                    instance_name = name.replace(f'{session}/', '').replace('-output', '')
                    missing.append(instance_name)
        
        return ready, missing


    def await_checkpoints(self, session, instance_names, poll_interval=0.1, timeout=300):
        '''Poll the datastore until all required checkpoints are available.
        
        This implements the "await" part of the promise pattern for eager fan-in.
        The fan-in function is invoked early, does as much work as possible,
        then calls this method to wait for any missing inputs.
        
        @param session The session ID  
        @param instance_names List of instance names whose checkpoints are needed
        @param poll_interval Seconds between polls (default 100ms)
        @param timeout Maximum seconds to wait before raising an exception
        @return True when all checkpoints are available
        @raises TimeoutError if timeout is exceeded
        '''
        start_time = time.time()
        
        while True:
            ready, missing = self.check_checkpoints_exist(session, instance_names)
            
            if not missing:
                if self.debug:
                    print(f'[DEBUG] All {len(instance_names)} checkpoints ready after {time.time() - start_time:.2f}s')
                return True
            
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f'Timeout waiting for checkpoints after {elapsed:.2f}s. '
                    f'Missing: {missing}'
                )
            
            if self.debug:
                print(f'[DEBUG] Waiting for {len(missing)} checkpoints: {missing[:3]}{"..." if len(missing) > 3 else ""}')
            
            time.sleep(poll_interval)


    def read_input_with_await(self, session, values, poll_interval=0.1, timeout=300):
        '''Read inputs for fan-in, waiting for any that aren't ready yet.
        
        This is a variant of read_input that supports the eager fan-in pattern.
        Instead of failing if inputs are missing, it polls until they're available.
        
        @param session The session ID
        @param values List of instance names to read
        @param poll_interval Seconds between polls for missing items
        @param timeout Maximum seconds to wait
        @return Ordered list of checkpoint data, same as read_input
        '''
        # First, wait for all checkpoints to exist
        self.await_checkpoints(session, values, poll_interval, timeout)
        
        # Then read them all (they should all exist now)
        return self.read_input(session, values)


    def create_lazy_inputs(self, session, instance_names, poll_interval=0.1, timeout=300):
        '''Create a LazyInputList for eager fan-in.
        
        Returns a list-like object where each element is a LazyInput that
        fetches data on-demand. This allows the fan-in function to execute
        as much as possible before blocking on inputs that aren't ready.
        
        @param session The session ID
        @param instance_names List of instance names for the fan-in inputs
        @param poll_interval Default poll interval for each LazyInput
        @param timeout Default timeout for each LazyInput
        @return LazyInputList containing LazyInput objects
        '''
        lazy_inputs = [
            LazyInput(
                datastore=self,
                session=session,
                instance_name=name,
                poll_interval=poll_interval,
                timeout=timeout,
                debug=self.debug
            )
            for name in instance_names
        ]
        
        return LazyInputList(lazy_inputs, debug=self.debug)


    def create_future_inputs(self, session, instance_names, ready_names=None, poll_interval=0.1, timeout=300):
        '''Create an AsyncFutureInputList for Future-Based execution (DynamoDB version).
        
        This is the preferred method for true async fan-in with asyncio.Event().
        The returned list supports both sync and async access patterns.
        
        @param session The session ID
        @param instance_names List of instance names for the fan-in inputs
        @param ready_names Optional list of names already known to be ready (from EagerFanIn metadata)
        @param poll_interval Default poll interval for each UnumFuture
        @param timeout Default timeout for each UnumFuture
        @return AsyncFutureInputList containing UnumFuture objects
        '''
        ready_set = set(ready_names) if ready_names else set()
        
        futures = []
        for name in instance_names:
            is_ready = name in ready_set
            
            if is_ready:
                # Pre-fetch the value since we know it's ready
                checkpoint = self.get_checkpoint_full(session, name)
                futures.append(UnumFuture(
                    datastore=self,
                    session=session,
                    instance_name=name,
                    value=checkpoint.get('User') if checkpoint else None,
                    is_ready=True,
                    poll_interval=poll_interval,
                    timeout=timeout,
                    debug=self.debug
                ))
            else:
                # Create a pending future
                futures.append(UnumFuture(
                    datastore=self,
                    session=session,
                    instance_name=name,
                    is_ready=False,
                    poll_interval=poll_interval,
                    timeout=timeout,
                    debug=self.debug
                ))
        
        return AsyncFutureInputList(futures, debug=self.debug)


    def get_checkpoint(self, session, instance_name):
        '''Given the session ID and the function's instance name, return the
        checkpoint's contents or None if the checkpoint doesn't exist.

        This function uses DynamoDB's GetItem API and request to read the
        `Value` field from the item.

        There doesn't seem to be a faster API to only check whether an item
        exists in DynamoDB without getting some of its attributes. GetItem
        seems to be the only API for this purpose.

        The GetItem operation returns the attributes requested in the
        ProjectionExpression for the item with the given primary key. If there
        is no matching item, GetItem does not return any data and there will
        be no Item element in the response.

        Example response:

        ```
        {
            'Item': {
                'string': 'string'|123|Binary(b'bytes')|True|None|set(['string'])|set([123])|set([Binary(b'bytes')])|[]|{}
            }
        }
        ```

        @session str
        @instance_name str
        '''
        try:
            ret = self.table.get_item(
                Key={
                    'Name': self.checkpoint_name(session, instance_name)
                },
                ConsistentRead=True
            )
        except Exception as e:
            print(f"[WARN] get_checkpoint() Error Code: {e.response['Error']['Code']}")
            raise e

        if "Item" in ret:
            item = ret["Item"]
            # Support both old format (Value) and new format (User)
            if "Value" in item:
                value = item["Value"]
            elif "User" in item:
                value = item["User"]
            else:
                return None
            # Parse JSON string if needed
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value
        else:
            return None


    def get_checkpoint_full(self, session, instance_name):
        '''Get the full checkpoint item including User data and GC info.
        
        Used by LazyInput to fetch a single input's data on-demand.
        
        @param session The session ID
        @param instance_name The instance name
        @return Dict with 'User' and optionally 'GC' keys, or None if not exists
        '''
        try:
            ret = self.table.get_item(
                Key={
                    'Name': self.checkpoint_name(session, instance_name)
                },
                ConsistentRead=True
            )
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] get_checkpoint_full() error: {e}")
            return None

        if "Item" not in ret:
            return None
        
        item = ret["Item"]
        result = {}
        
        # Parse the User field (stored as JSON string)
        if 'User' in item:
            result['User'] = json.loads(item['User']) if isinstance(item['User'], str) else item['User']
        elif 'Value' in item:
            # Fallback for older format
            result['User'] = json.loads(item['Value']) if isinstance(item['Value'], str) else item['Value']
        
        # Get GC info if present
        if 'GC' in item:
            result['GC'] = item['GC']
        
        return result



    def _create_if_not_exist(self, key_name, key, value):
        '''Create an item in the DynamoDB table with primary key `key` and
        content `value` if the key does not already exist

        @return a positive integer if success. -1 if the key already exists.
        '''
        item = {key_name: key, **value}
        try:
            if self.debug:
                rsp = self.table.put_item(Item=item,
                    ConditionExpression='attribute_not_exists(#N)',
                    ExpressionAttributeNames={"#N": key_name},
                    ReturnConsumedCapacity='TOTAL'
                )

                return int(rsp['ConsumedCapacity']['CapacityUnits'])

            else:
                self.table.put_item(Item=item,
                    ConditionExpression='attribute_not_exists(#N)',
                    ExpressionAttributeNames={"#N": key_name}

                )
                return 1

        except ClientError as e:
            if e.response['Error']['Code']=='ConditionalCheckFailedException':
                return -1
            elif e.response['Error']['Code']=='ValidationException':
                raise e
            else:
                raise e
        except Exception as e:
            # print(f"[WARN] Error Code is {e.response['Error']['Code']}")
            raise e



    def checkpoint_name(self, session, instance_name):
        '''Given the session ID and instance name, return the name of its
        DynamoDB checkpoint
        '''
        return f'{session}/{instance_name}-output'



    def checkpoint(self, session, instance_name, data):
        '''Writing the user function output as an item with a unique name

        This function creates a single item in the dynamoDB table. The item
        contains the user function output of this particular function
        instance.

        Items have the following schema:

        ```
        {
            "Session": "a uuid4 string",
            "Name": "<session>/<instance_name>-output",
            "Value": "function result as a JSON string"
        }
        ```

        The "Name" field is the primary key.

        The "Value" field is of type string and is
        json.dumps(data)

        This function will only try to write if an item with the same "Name"
        does NOT already exists. If an item with the same "Name" already
        exists, the DynamoDB PutItem call is called and this function returns
        1.

        If the data to write failed DynamoDB's schema validation, return 2.
        '''

        return self._create_if_not_exist("Name", self.checkpoint_name(session, instance_name), data)



    def _delete(self, key_name, key):
        '''Delete a key

        Delete is idempotent. Deleting the same key multiple times does not
        raise an exception. Similarly, if the key does not exist, _delete does
        not raise an exception.

        @return the consumed capacity
        '''
        try:
            if self.debug:
                rsp = self.table.delete_item(
                    Key={key_name: key},
                    ReturnConsumedCapacity='TOTAL')

                return int(rsp['ConsumedCapacity']['CapacityUnits'])
            else:
                rsp = self.table.delete_item(Key={key_name: key})
                return 1

        except ClientError as e:
            raise e



    def delete_checkpoint(self, session, instance_name):
        return self._delete("Name", self.checkpoint_name(session, instance_name))



    def gc_sync_point_name(self, session, parent_function_instance_name):

        return f'{session}/{parent_function_instance_name}-gc'



    def fanin_sync_point_name(self, session, aggregation_function_instance_name):

        return f'{session}/{aggregation_function_instance_name}-fanin'



    def gc_sync_ready(self, session, parent_function_instance_name, index, num_branches):
        '''Mark my gc as ready and check if gc is ready to run

        In the case of a parent node invoking multiple downstream child nodes,
        all child nodes need to have created their checkpoints before the
        parent node's checkpoint can be garbage collected. Unum have all child
        nodes synchronize via the intermediate datastore so that the
        last-to-finish child node deletes the parent's checkpoint.

        The synchronization item is named by the gc_sync_point_name() function
        based on the session ID and the parent function's instance name.

        @return True is I'm the last-to-finish child node. False if I'm not.
        '''

        return self._sync_ready(self.gc_sync_point_name(session, parent_function_instance_name), index, num_branches)



    def fanin_sync_ready(self, session, aggregation_function_instance_name, index, my_instance_name, num_branches):
        '''Mark my branch as ready and check if fan-in is ready to run

        In the case of fan-in, all upstream branches need to have created
        their checkpoints before the aggregation function is invoked. Unum
        have all branches synchronize via the intermediate datastore so that only
        the last-to-finish branch invokes the aggregation function.

        The synchronization item is named by the fanin_sync_point_name() function
        based on the session ID and the aggregation function's instance name.

        @return True is I'm the last-to-finish branch. False if I'm not.
        '''

        return self._sync_ready(self.fanin_sync_point_name(session, aggregation_function_instance_name), index, num_branches)



    def _sync_ready(self, sync_point_name, index, num_branches):
        '''Mark the caller ready and return the return map

        @param sync_point_name DynamoDB primary key of the synchronization item
        @param index caller's index in the synchronization item
        @param num_branches the number of nodes that need to synchronize
        '''
        return self._sync_ready_bitmap(sync_point_name, index, num_branches)



    def _sync_ready_bitmap(self, sync_point_name, index, num_branches):
        self._create_bitmap(sync_point_name, num_branches)
        ready_map = self._update_bitmap_result(sync_point_name, index)

        return self._bitmap_ready(ready_map)



    def _create_bitmap(self, bitmap_name, bitmap_size):

        value = {"ReadyMap": [False for i in range(bitmap_size)]}

        return self._create_if_not_exist("Name", bitmap_name, value)



    def _update_bitmap_result(self, bitmap_name, index):
        try:
            ret = self.table.update_item(
                Key={"Name": bitmap_name},
                ReturnValues='ALL_NEW',
                UpdateExpression="set #L[" + str(index) + "] = :nd",
                ConditionExpression='attribute_exists(#N)',
                ExpressionAttributeValues={':nd': True},
                ExpressionAttributeNames={"#N": "Name", "#L": "ReadyMap"})
        except Exception as e:
            raise e

        return ret['Attributes']['ReadyMap']



    def _bitmap_ready(self, bitmap):
        '''Check if the bitmap is all True
        '''

        for b in bitmap:
            if b == False:
                return False
        return True



    def _sync_ready_counter(self, sync_point_name, index, num_branches):
        pass



    def _update_fan_in_counter(self, session, counter_name):
        '''Given the session and counter name, create the counter with initial
        0 if it does not already exist. Or increments the counter by 1
        atomically. Return the counter value after update.

        According to
        https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/WorkingWithItems.html#WorkingWithItems.AtomicCounters,
        UpdateItem with SET on numerical values are guaranteed to be atomic.
        '''
        try:
            ret = self.table.put_item(Item={
                    "Name": f'{session}/{counter_name}',
                    "Count": 0
                },
                ConditionExpression='attribute_not_exists(#N)',
                ExpressionAttributeNames={"#N": "Name"}

            )
        except ClientError as e:  
            if e.response['Error']['Code']=='ConditionalCheckFailedException':  
                pass
            else:
                raise e
        except Exception as e:
            raise e


        try:
            ret = self.table.update_item(
                Key={"Name": f'{session}/{counter_name}'},
                ReturnValues='UPDATED_NEW',
                UpdateExpression='SET #C = #C + :incr',
                ConditionExpression='attribute_exists(#N)',
                # ExpressionAttributeNames={
                #     'string': 'string'
                # },
                ExpressionAttributeValues={':incr': 1},
                ExpressionAttributeNames={"#N": "Name", "#C": 'Count'})
        except Exception as e:
            raise e

        return ret["Attributes"]["Count"]



    def check_fan_in_complete(self, session, values, target_count):
        '''Increment the counter and check if fan-in is complete

        Fan-in with DynamoDB is considered complete when the counter number
        equals the fan-out size.
        '''
        counter_name = ""
        for v in values:
            counter_name = f'{counter_name}{v}-'
        counter_name = f'{counter_name}counter'

        ret = self._update_fan_in_counter(session, counter_name)

        return ret == target_count



class S3Driver(UnumIntermediaryDataStore):
    def __init__(self, ds_name):
        ''' Initialze an s3 data store

        Raise an exception if the bucket doesn't exist.

        @ param ds_name an s3 bucket name
        '''
        super(S3Driver, self).__init__("s3", ds_name)
        self.backend = boto3.client("s3")
        # check if this bucket exists and this function has permission to
        # access it
        try:
            response = self.backend.head_bucket(Bucket=self.name)
        except:
            raise IOError(f'The intermediary s3 bucket does NOT exist')


    def create_session(self):
        ''' Create a prefix (directory) in the bucket
        '''
        return f'{uuid.uuid4()}'

    def create_fanin_context(self):
        ''' For the fan-out functions to write their outputs, creates a s3
        directory
        DEPRECATED
        '''
        directoryName = f'{uuid.uuid4()}'
        self.backend.put_object(Bucket=self.name, Key=(directoryName+'/'))

        return directoryName

    def read_input(self, session, values):
        '''Given the workflow invocation session id and a list of pointers to
        the intermediary data store, read all data and return them as an
        ordered list.

        Data in the returned list should correspond to the pointers in the
        `values` parameter _in the same order_.

        In practice, this function is used by the fan-in function to read its
        input, which is the outputs of all fan-out functions, from the
        intermediary data store.

        Each element in the `values` list combined with `session` (e.g.,
        "{session}/{values[0]}-output.json") is a key in the intermediary s3
        bucket.

        The pointers are used as is. It is the invoker's responsibility to
        expand the pointers and make sure that they are valid. unum no longer
        uses glob patterns in the the runtime payload (but the unum config
        language still supports glob patterns) and the invoker should expand
        all glob patterns to concrete data pointer names.

        unum guarantees that all pointers in `values` exists when this
        function is called (i.e., when the fan-in function is invoked).
        Therefore, if one of the pointers doesn't exist in the data store,
        this function will throw an exception.

        On AWS, there's no reason to pass in a single data pointer when
        invoking a Lambda because asynchronou HTTP requests achieves the same
        results by adding the data onto Lambda's event queue. Therefore, the
        `values` parameter should always be a list. We don't consider the
        scenario where `values` is a dict.
        '''
        s3_names = [f'{session}/{p}-output.json' for p in ptr]

        data = []

        for s3_name, p in zip(s3_names, ptr):
            local_file_name = f'{p}-output.json'
            self.backend.download_file(self.name, s3_name, f'/tmp/{local_file_name}')

            with open(f'/tmp/{local_file_name}', 'r') as f:
                data.append(json.loads(f.read()))

        return data

    def check_value_exist(self, session, name):
        pass


    def check_values_exist(self, session, names):

        s3_names = [f'{session}/{n}-output.json' for n in names]

        response = self.backend.list_objects_v2(
                        Bucket=self.name,
                        Prefix=f'{session}/' # e.g., reducer0/
                    )
        all_keys = [e["Key"] for e in response["Contents"]]

        for n in s3_names:
            if n not in all_keys:
                return False

        return True


    def write_error(self, session, name, msg):
        ''' Save an error message
        @session
        @name str name of the s3 file
        @msg json-serializable

        '''
        local_file_path = f'/tmp/{name}'
        with open(local_file_path, 'w') as f:
            f.write(json.dumps(msg))

        self.backend.upload_file(local_file_path,
                                 self.name,
                                 f'{session}/{name}')

    def write_return_value(self, session, ret_name, ret):
        ''' Write a user function's return value to the s3 bucket

        @param session a s3 prefix that is the session context
        @param ret_name the s3 file name
        @param ret the user function's return value
        '''
        fn = f'{ret_name}-output.json'
        local_file_path = '/tmp/'+fn
        with open(local_file_path, 'w') as f:
            f.write(json.dumps(ret))

        self.backend.upload_file(local_file_path,
                                 self.name,
                                 f'{session}/{fn}')

    def write_fanin_context(self, output, fcn_name, context, index, size):
        ''' Fan-out function writes its outputs to the fan-in s3 directory

            @param output function output
            @param fcn_name lambda function's name
            @param context s3 directory name (without the /)
            @param index function's index in the fan-out
            @param size fan-out size
            DEPRECATED
        '''
        fn = f"{fcn_name}-UINDEX-{index}-outof-{size}.json"
        local_file_path = '/tmp/'+fn

        with open(local_file_path, 'w') as f:
            f.write(json.dumps(output))

        self.backend.upload_file(local_file_path,
                                 self.name,
                                 f'{context}/{fn}')

    def get_index(self, fn):
        s = fn.split("UINDEX")[1]
        return s.split("-")[1]

    def check_prefix_index_exist(self, context, prefix, index):
        file_list = self.list_fanin_context(context)
        file_list = [e.replace(f'{context}/',""), file_list]
        target_list = list(filter(lambda x : x.startswith(prefix), l))
        for p in target_list:
            if self.get_index(p) == index:
                return True

        return False

    def list_fanin_context(self, context):
        ''' List all the files in the s3 fan-in directory
        '''
        response = self.backend.list_objects(
                        Bucket=self.name,
                        Prefix=f'{context}/' # e.g., reducer0/
                    )

        keys = list(filter(lambda x: x.endswith('/') == False, [e['Key'] for e in response['Contents']]))

        return keys

    def read_fanin_context(self, context, keys=None):
        ''' Read all files in the fan-in directory and return it as an ordered
        list
        '''
        response = self.backend.list_objects(
            Bucket=self.name,
            Prefix=f"{context}/" # e.g., reducer0/
            )

        file_list = [e['Key'] for e in response['Contents']]
           
        os.makedirs(f"/tmp/{context}", exist_ok = True)

        if keys != None:
            file_list = filter(lambda x : x in keys, file_list)

        for k in file_list:
            if k.endswith('/'):
                continue

            self.backend.download_file(self.name, k, f"/tmp/{k}")

        # return data as a list
        ret = []
        fl = os.listdir(f"/tmp/{context}/")
        fnc = fl[0].split('UINDEX')
        prefix = f'{fnc[0]}UINDEX'
        tmp = fnc[1].split('-')
        suffix = f'{tmp[2]}-{tmp[3]}'

        for i in range(len(fl)):

            with open(f'/tmp/{context}/{prefix}-{i}-{suffix}', 'r') as f:
                ret.append(json.loads(f.read()))

        return ret


# =============================================================================
# MODULE-LEVEL STREAMING FUNCTIONS
# These functions provide a simple interface for the unum_streaming module
# to read/write streaming field values without needing a full datastore instance.
# =============================================================================

_streaming_ds_instance = None
_streaming_ds_initialized = False

def _get_streaming_datastore():
    """Get or create a datastore instance for streaming operations."""
    global _streaming_ds_instance, _streaming_ds_initialized
    
    if _streaming_ds_initialized:
        return _streaming_ds_instance
    
    _streaming_ds_initialized = True
    
    try:
        ds_type = os.environ.get('UNUM_INTERMEDIARY_DATASTORE_TYPE', 'dynamodb')
        ds_name = os.environ.get('UNUM_INTERMEDIARY_DATASTORE_NAME', '')
        
        if not ds_name:
            print('[Streaming] No UNUM_INTERMEDIARY_DATASTORE_NAME set, streaming disabled')
            return None
        
        _streaming_ds_instance = UnumIntermediaryDataStore.create(ds_type, ds_name, False)
    except Exception as e:
        print(f'[Streaming] Failed to initialize datastore: {e}')
        _streaming_ds_instance = None
    
    return _streaming_ds_instance


def write_intermediary(key: str, value: str) -> bool:
    """
    Write a value to the intermediary datastore for streaming.
    
    This is a module-level convenience function used by unum_streaming.
    
    Args:
        key: The key to write to (format: streaming/{session}/{source}/{field})
        value: The JSON-serialized value to store
        
    Returns:
        True if successful
    """
    ds = _get_streaming_datastore()
    if ds is None:
        return False
    
    try:
        if ds.my_type == 'dynamodb':
            # Use DynamoDB table directly
            ds.table.put_item(
                Item={
                    "Name": key,
                    "Value": value,
                    "Timestamp": datetime.datetime.now().isoformat()
                }
            )
        elif ds.my_type == 's3':
            # Use S3 - write to temp file first
            local_path = f'/tmp/{key.replace("/", "_")}'
            with open(local_path, 'w') as f:
                f.write(value)
            ds.backend.upload_file(local_path, ds.name, key)
        else:
            print(f'[Streaming] Unsupported datastore type: {ds.my_type}')
            return False
        return True
    except Exception as e:
        print(f'[Streaming] Error writing to datastore: {e}')
        return False


def read_intermediary(key: str) -> Optional[str]:
    """
    Read a value from the intermediary datastore for streaming.
    
    This is a module-level convenience function used by unum_streaming.
    
    Args:
        key: The key to read from (format: streaming/{session}/{source}/{field})
        
    Returns:
        The value as a string, or None if not found
    """
    ds = _get_streaming_datastore()
    if ds is None:
        return None
    
    try:
        if ds.my_type == 'dynamodb':
            response = ds.table.get_item(
                Key={"Name": key},
                ConsistentRead=True
            )
            item = response.get('Item')
            if item:
                return item.get('Value')
            return None
        elif ds.my_type == 's3':
            local_path = f'/tmp/{key.replace("/", "_")}'
            try:
                ds.backend.download_file(ds.name, key, local_path)
                with open(local_path, 'r') as f:
                    return f.read()
            except:
                return None
        else:
            print(f'[Streaming] Unsupported datastore type: {ds.my_type}')
            return None
    except Exception as e:
        # Key not found or other error
        return None


def delete_intermediary(key: str) -> bool:
    """
    Delete a value from the intermediary datastore.
    
    Used for cleanup of streaming keys after workflow completes.
    
    Args:
        key: The key to delete (format: streaming/{session}/{source}/{field})
        
    Returns:
        True if successful
    """
    ds = _get_streaming_datastore()
    if ds is None:
        return False
    
    try:
        if ds.my_type == 'dynamodb':
            ds.table.delete_item(Key={"Name": key})
        elif ds.my_type == 's3':
            ds.backend.delete_object(Bucket=ds.name, Key=key)
        else:
            print(f'[Streaming] Unsupported datastore type: {ds.my_type}')
            return False
        return True
    except Exception as e:
        print(f'[Streaming] Error deleting from datastore: {e}')
        return False


def list_streaming_keys(session_id: str) -> list:
    """
    List all streaming keys for a given session.
    
    Used for cleanup after workflow completes.
    
    Args:
        session_id: The session ID to list keys for
        
    Returns:
        List of keys matching the pattern streaming/{session_id}/*
    """
    ds = _get_streaming_datastore()
    if ds is None:
        return []
    
    prefix = f"streaming/{session_id}/"
    keys = []
    
    try:
        if ds.my_type == 'dynamodb':
            # Scan for keys with prefix (not ideal but simple)
            response = ds.table.scan(
                FilterExpression="begins_with(#n, :prefix)",
                ExpressionAttributeNames={"#n": "Name"},
                ExpressionAttributeValues={":prefix": prefix}
            )
            for item in response.get('Items', []):
                keys.append(item.get('Name'))
        elif ds.my_type == 's3':
            response = ds.backend.list_objects_v2(Bucket=ds.name, Prefix=prefix)
            for obj in response.get('Contents', []):
                keys.append(obj.get('Key'))
    except Exception as e:
        print(f'[Streaming] Error listing keys: {e}')
    
    return keys


def cleanup_session_streaming_keys(session_id: str) -> int:
    """
    Clean up all streaming keys for a completed session.
    
    Should be called at the end of a workflow to remove
    temporary streaming data.
    
    Args:
        session_id: The session ID to clean up
        
    Returns:
        Number of keys deleted
    """
    keys = list_streaming_keys(session_id)
    deleted = 0
    
    for key in keys:
        if delete_intermediary(key):
            deleted += 1
    
    if deleted > 0:
        print(f'[Streaming] Cleaned up {deleted} streaming keys for session {session_id}')
    
    return deleted


