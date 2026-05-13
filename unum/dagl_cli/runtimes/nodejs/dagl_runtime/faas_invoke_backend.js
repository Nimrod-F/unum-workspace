/**
 * DAGL Runtime Layer - FaaS Invocation Backend (Node.js)
 *
 * Supports multiple platforms:
 *   - aws: AWS Lambda async invoke (SDK v3)
 *   - http: Direct HTTP POST to function URL (cross-platform)
 *   - multi: Router that dispatches per-function based on platform field
 *
 * DAGL_FUNCTION_MAP format:
 *   Legacy (single-platform):  {"FuncName": "arn:aws:lambda:..."}
 *   New (multi-platform):      {"FuncName": {"platform": "aws", "arn": "...", "url": "..."}}
 */

'use strict';

const { LambdaClient, InvokeCommand } = require('@aws-sdk/client-lambda');

/**
 * Load function map from env var.
 */
function loadFunctionMap() {
  const mapStr = process.env.DAGL_FUNCTION_MAP || '{}';
  return JSON.parse(mapStr);
}

/**
 * Check if function map uses multi-platform dict format.
 */
function isMultiPlatformMap(mapping) {
  for (const v of Object.values(mapping)) {
    if (typeof v === 'object' && v !== null) return true;
  }
  return false;
}

// ─── AWS Lambda Backend ──────────────────────────────────────────────────────────

class AWSLambdaBackend {
  constructor() {
    this.lambdaClient = new LambdaClient({});
    this.mapping = loadFunctionMap();
  }

  async invoke(functionName, data) {
    const target = this.mapping[functionName];
    const arn = typeof target === 'string' ? target : (target.arn || target.url);
    if (!arn) {
      throw new Error(`DAGL AWS: No ARN mapping for function '${functionName}'.`);
    }
    await this._invokeAsync(arn, data);
  }

  async invokeArn(arn, data) {
    await this._invokeAsync(arn, data);
  }

  async _invokeAsync(functionArn, data) {
    const command = new InvokeCommand({
      FunctionName: functionArn,
      InvocationType: 'Event',
      LogType: 'None',
      Payload: Buffer.from(JSON.stringify(data)),
    });
    await this.lambdaClient.send(command);
  }
}

// ─── HTTP Backend (cross-platform, zero external deps) ───────────────────────────

class HTTPInvocationBackend {
  constructor() {
    this.mapping = loadFunctionMap();
  }

  async invoke(functionName, data) {
    const target = this.mapping[functionName];
    const url = typeof target === 'string' ? target : (target && target.url);
    if (!url) {
      throw new Error(`DAGL HTTP: No URL for function '${functionName}'.`);
    }
    await this.invokeUrl(url, data);
  }

  async invokeUrl(url, data) {
    await this._postAsync(url, data);
  }

  async _postAsync(url, data) {
    const payload = JSON.stringify(data);
    const parsedUrl = new URL(url);
    const mod = parsedUrl.protocol === 'https:' ? require('https') : require('http');

    return new Promise((resolve, reject) => {
      const req = mod.request(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(payload),
        },
        timeout: 10000,
      }, (res) => {
        // Consume response to free socket
        res.resume();
        res.on('end', resolve);
      });

      req.on('error', (e) => {
        console.log(`[DAGL HTTP] Warning: POST to ${url} failed: ${e.message}`);
        resolve(); // Don't crash — downstream may still process
      });

      req.on('timeout', () => {
        req.destroy();
        console.log(`[DAGL HTTP] Warning: POST to ${url} timed out`);
        resolve();
      });

      req.write(payload);
      req.end();
    });
  }
}

// ─── Multi-Platform Router Backend ───────────────────────────────────────────────

class MultiPlatformBackend {
  /**
   * Routes invocations per-function based on platform field in function map.
   *
   * Routing rules:
   *   - Same platform → native SDK (faster, internal network)
   *   - Different platform → HTTP POST to function URL (cross-cloud)
   */
  constructor() {
    this.mapping = loadFunctionMap();
    this.myPlatform = process.env.FAAS_PLATFORM_NATIVE ||
                      process.env.FAAS_PLATFORM || 'aws';

    // Lazy-init backends
    this._awsBackend = null;
    this._httpBackend = null;
  }

  get awsBackend() {
    if (!this._awsBackend) {
      this._awsBackend = new LambdaClient({});
    }
    return this._awsBackend;
  }

  get httpBackend() {
    if (!this._httpBackend) {
      this._httpBackend = new HTTPInvocationBackend();
    }
    return this._httpBackend;
  }

  async invoke(functionName, data) {
    const target = this.mapping[functionName];
    if (!target) {
      throw new Error(`DAGL Multi: No mapping for function '${functionName}'.`);
    }

    // Legacy format (string) — treat as same-platform
    if (typeof target === 'string') {
      if (this.myPlatform === 'aws') {
        await this._invokeAwsNative(target, data);
      } else {
        await this.httpBackend.invokeUrl(target, data);
      }
      return;
    }

    // New format (dict with platform field)
    const targetPlatform = target.platform || 'aws';

    if (targetPlatform === this.myPlatform) {
      // Same platform → native invocation
      await this._invokeNative(targetPlatform, target, data);
    } else {
      // Cross-platform → prefer direct SDK invocation if possible
      if (targetPlatform === 'aws' && target.arn) {
        await this._invokeAwsNative(target.arn, data, target.region);
      } else {
        const url = target.url;
        if (!url) {
          throw new Error(
            `DAGL Multi: Cross-platform call to '${functionName}' requires a URL or ARN. ` +
            `Target: ${targetPlatform}, My: ${this.myPlatform}`);
        }
        await this.httpBackend.invokeUrl(url, data);
      }
    }
  }

  async _invokeNative(platform, target, data) {
    if (platform === 'aws') {
      const arn = target.arn;
      if (arn) {
        await this._invokeAwsNative(arn, data);
      } else {
        await this.httpBackend.invokeUrl(target.url, data);
      }
    } else if (platform === 'gcp') {
      // GCP native = HTTP POST to function URL (Cloud Functions v2)
      const url = target.url;
      if (url) {
        await this.httpBackend.invokeUrl(url, data);
      } else {
        throw new Error('DAGL Multi: GCP target requires URL');
      }
    } else {
      const url = target.url;
      if (url) {
        await this.httpBackend.invokeUrl(url, data);
      } else {
        throw new Error(`DAGL Multi: unsupported platform '${platform}' and no URL`);
      }
    }
  }

  async _invokeAwsNative(functionArn, data, region) {
    const command = new InvokeCommand({
      FunctionName: functionArn,
      InvocationType: 'Event',
      LogType: 'None',
      Payload: Buffer.from(JSON.stringify(data)),
    });
    if (region) {
      const client = new LambdaClient({ region });
      await client.send(command);
    } else {
      await this.awsBackend.send(command);
    }
  }
}

// ─── Backend Factory ─────────────────────────────────────────────────────────────

/**
 * Create the appropriate invocation backend based on platform and function map.
 * @param {string} platform - 'aws', 'gcloud', 'http', 'multi', or 'fake'
 * @returns {AWSLambdaBackend|HTTPInvocationBackend|MultiPlatformBackend}
 */
function createBackend(platform) {
  // Auto-detect: if function map has mixed platforms, use multi
  if (platform === 'multi' || platform === 'aws') {
    const mapping = loadFunctionMap();
    if (isMultiPlatformMap(mapping)) {
      return new MultiPlatformBackend();
    }
  }

  switch (platform) {
    case 'aws': return new AWSLambdaBackend();
    case 'http': return new HTTPInvocationBackend();
    case 'multi': return new MultiPlatformBackend();
    default:
      throw new Error(`DAGL: Unsupported platform '${platform}'`);
  }
}

module.exports = {
  AWSLambdaBackend,
  HTTPInvocationBackend,
  MultiPlatformBackend,
  createBackend,
  loadFunctionMap,
  isMultiPlatformMap,
};
