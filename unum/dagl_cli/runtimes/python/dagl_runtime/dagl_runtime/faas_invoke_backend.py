"""DAGL Runtime Layer - FaaS Invocation Backend

Reads function mapping from DAGL_FUNCTION_MAP env var.

Supports multiple platforms:
  - aws: AWS Lambda async invoke (boto3)
  - gcloud: GCP Cloud Functions via Pub/Sub
  - http: Direct HTTP POST to function URL (cross-platform)
  - multi: Router that dispatches per-function based on platform field in mapping
  - fake: Local testing (prints payload)

DAGL_FUNCTION_MAP format:
  Legacy (single-platform):  {"FuncName": "arn:aws:lambda:..."}
  New (multi-platform):      {"FuncName": {"platform": "aws", "arn": "...", "url": "..."}}
"""

import json
import os


class InvocationBackend(object):
    subclasses = {}

    @classmethod
    def add_backend(cls, platform):
        def wrapper(subclass):
            cls.subclasses[platform] = subclass
            return subclass
        return wrapper

    @classmethod
    def create(cls, platform):
        if platform not in cls.subclasses:
            raise ValueError(f'DAGL runtime does not support platform: {platform}')
        return cls.subclasses[platform]()


def _load_function_map():
    """Load and return the function map from env var."""
    map_str = os.environ.get('DAGL_FUNCTION_MAP', '{}')
    return json.loads(map_str)


def _is_multi_platform_map(mapping):
    """Check if function map uses the new multi-platform dict format."""
    for v in mapping.values():
        if isinstance(v, dict):
            return True
    return False


# ─── AWS Lambda Backend ──────────────────────────────────────────────────────────

@InvocationBackend.add_backend('aws')
class AWSLambdaBackend(InvocationBackend):

    def __init__(self):
        import boto3
        self.lambda_client = boto3.client("lambda")
        
        # Read function mapping from env var (set by dagl deploy)
        self.mapping = _load_function_map()

    def invoke(self, function, data):
        target = self.mapping[function]
        # Support both legacy (string) and new (dict) format
        arn = target if isinstance(target, str) else target.get('arn', target.get('url'))
        return self._http_invoke_async(arn, data)

    def invoke_arn(self, arn, data):
        """Invoke a specific ARN directly (used by MultiPlatformBackend)."""
        return self._http_invoke_async(arn, data)

    def _http_invoke_async(self, function_arn, data):
        """Invoke a Lambda function asynchronously (fire-and-forget)."""
        response = self.lambda_client.invoke(
            FunctionName=function_arn,
            InvocationType='Event',
            LogType='None',
            Payload=json.dumps(data),
        )
        response['Payload'].read()
        return


# ─── GCP Cloud Functions Backend (Pub/Sub) ───────────────────────────────────────

@InvocationBackend.add_backend('gcloud')
class GCloudFunctionBackend(InvocationBackend):

    def __init__(self):
        from google.cloud import pubsub_v1
        self.pubsub = pubsub_v1.PublisherClient()
        
        # Read function mapping from env var
        self.mapping = _load_function_map()

    def invoke(self, function, data):
        target = self.mapping[function]
        topic = target if isinstance(target, str) else target.get('topic')
        self._pubsub_invoke(topic, data)

    def _pubsub_invoke(self, topic, data):
        try:
            self.pubsub.publish(topic, json.dumps(data).encode('utf-8'))
        except Exception as e:
            raise e


# ─── HTTP Backend (cross-platform, zero deps) ────────────────────────────────────

@InvocationBackend.add_backend('http')
class HTTPInvocationBackend(InvocationBackend):
    """Invoke functions via HTTP POST to their URL endpoint.
    
    Used for cross-platform calls (e.g., AWS → GCP, GCP → AWS).
    Works with:
      - AWS Lambda Function URLs
      - GCP Cloud Functions v2 HTTP triggers
      - Any HTTP endpoint that accepts JSON POST
    
    Fire-and-forget: sends POST, checks for 2xx, does not wait for response body.
    """

    def __init__(self):
        self.mapping = _load_function_map()

    def invoke(self, function, data):
        target = self.mapping[function]
        url = target if isinstance(target, str) else target.get('url')
        if not url:
            raise ValueError(f'DAGL HTTP backend: no URL for function "{function}". '
                             f'Mapping: {target}')
        self._post_async(url, data)

    def invoke_url(self, url, data):
        """Post to a specific URL directly (used by MultiPlatformBackend)."""
        self._post_async(url, data)

    def _post_async(self, url, data):
        """HTTP POST JSON payload to URL. Fire-and-forget."""
        import urllib.request
        
        payload = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                # Read response to release connection, but don't process it
                resp.read()
        except Exception as e:
            # Log but don't crash — downstream function may still process the request
            print(f'[DAGL HTTP] Warning: POST to {url} failed: {e}')


# ─── Multi-Platform Router Backend ───────────────────────────────────────────────

@InvocationBackend.add_backend('multi')
class MultiPlatformBackend(InvocationBackend):
    """Routes invocations per-function based on platform field in function map.
    
    Function map format (new):
    {
      "Tokenize": {"platform": "aws", "arn": "arn:aws:lambda:...", "url": "https://..."},
      "Analyze":  {"platform": "gcp", "url": "https://europe-west1-project.cloudfunctions.net/..."},
      "Report":   {"platform": "aws", "arn": "arn:aws:lambda:...", "url": "https://..."}
    }
    
    Routing rules:
      - If target platform == my platform → use native SDK (fast, internal network)
      - If target platform != my platform → use HTTP POST to function URL (cross-cloud)
    """

    def __init__(self):
        self.mapping = _load_function_map()
        self.my_platform = os.environ.get('FAAS_PLATFORM_NATIVE', 
                                          os.environ.get('FAAS_PLATFORM', 'aws'))
        
        # Lazy-init native backends (only when first needed)
        self._aws_backend = None
        self._gcp_backend = None
        self._http_backend = None

    @property
    def aws_backend(self):
        if self._aws_backend is None:
            import boto3
            self._aws_backend = boto3.client("lambda")
        return self._aws_backend

    @property
    def http_backend(self):
        if self._http_backend is None:
            self._http_backend = HTTPInvocationBackend()
        return self._http_backend

    def invoke(self, function, data):
        target = self.mapping.get(function)
        if target is None:
            raise ValueError(f'DAGL Multi: no mapping for function "{function}"')
        
        # Legacy format (string) — treat as same-platform
        if isinstance(target, str):
            if self.my_platform == 'aws':
                self._invoke_aws_native(target, data)
            else:
                self.http_backend.invoke_url(target, data)
            return
        
        # New format (dict with platform field)
        target_platform = target.get('platform', 'aws')
        
        if target_platform == self.my_platform:
            # Same platform → native invocation (faster, internal network)
            self._invoke_native(target_platform, target, data)
        else:
            # Cross-platform → use native SDK of the target if possible
            if target_platform == 'aws' and target.get('arn'):
                # Call AWS Lambda directly using boto3 (works from any platform)
                self._invoke_aws_native(target['arn'], data, region=target.get('region'))
            else:
                # HTTP POST to function URL
                url = target.get('url')
                if not url:
                    raise ValueError(
                        f'DAGL Multi: cross-platform call to "{function}" requires a URL or ARN. '
                        f'Target platform: {target_platform}, my platform: {self.my_platform}')
                self.http_backend.invoke_url(url, data)

    def _invoke_native(self, platform, target, data):
        """Invoke using native platform SDK."""
        if platform == 'aws':
            arn = target.get('arn')
            if arn:
                self._invoke_aws_native(arn, data)
            else:
                # Fallback to URL
                self.http_backend.invoke_url(target['url'], data)
        elif platform == 'gcp':
            # On GCP, native = HTTP POST to function URL (Cloud Functions v2)
            # Pub/Sub is an option but HTTP is simpler for direct invocation
            url = target.get('url')
            if url:
                self.http_backend.invoke_url(url, data)
            else:
                raise ValueError(f'DAGL Multi: GCP target requires URL')
        else:
            # Unknown platform — try URL
            url = target.get('url')
            if url:
                self.http_backend.invoke_url(url, data)
            else:
                raise ValueError(f'DAGL Multi: unsupported platform "{platform}" and no URL')

    def _invoke_aws_native(self, function_arn, data, region=None):
        """AWS Lambda async invoke via SDK."""
        if region:
            import boto3
            client = boto3.client("lambda", region_name=region)
        else:
            client = self.aws_backend
        response = client.invoke(
            FunctionName=function_arn,
            InvocationType='Event',
            LogType='None',
            Payload=json.dumps(data),
        )
        response['Payload'].read()


# ─── Fake Backend (testing) ──────────────────────────────────────────────────────

@InvocationBackend.add_backend('fake')
class FakeFaaSBackend(InvocationBackend):

    def invoke(self, function, data):
        print(f'[DAGL FaaS: fake] Invoking {function}')
        print(f'[DAGL FaaS: fake] Payload: {data}')
        return data
