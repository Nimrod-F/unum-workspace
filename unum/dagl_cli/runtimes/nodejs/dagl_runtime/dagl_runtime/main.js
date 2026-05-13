/**
 * DAGL Runtime Layer - Lambda Entry Point (Node.js)
 *
 * This module replaces the user's Lambda handler. It:
 * 1. Reads the user's original handler from DAGL_USER_HANDLER env var
 * 2. Reads the orchestration config from DAGL_CONFIG env var
 * 3. Reads the function ARN mapping from DAGL_FUNCTION_MAP env var
 * 4. Wraps the user function with DAGL orchestration
 *
 * Lambda Handler: dagl_runtime/main.handler
 */

'use strict';

const { Unum } = require('./unum');
const { randomUUID } = require('crypto');

// ── Dynamic user handler import ─────────────────────────────────────────────────

function _importUserHandler() {
  const handlerPath = process.env.DAGL_USER_HANDLER;
  if (!handlerPath) {
    throw new Error("DAGL_USER_HANDLER env var not set. Set it to your original handler (e.g., 'index.handler')");
  }

  const parts = handlerPath.split('.');
  if (parts.length < 2) {
    throw new Error(`Invalid DAGL_USER_HANDLER format: '${handlerPath}'. Expected 'module.function' (e.g., 'index.handler')`);
  }

  const functionName = parts.pop();
  const modulePath = parts.join('.');

  // Lambda puts user code in LAMBDA_TASK_ROOT (default: /var/task)
  const taskRoot = process.env.LAMBDA_TASK_ROOT || '/var/task';
  let userModule;
  try {
    userModule = require(`${taskRoot}/${modulePath}`);
  } catch (e) {
    throw new Error(`Cannot import user handler module '${modulePath}': ${e.message}`);
  }

  const handlerFn = userModule[functionName];
  if (!handlerFn) {
    throw new Error(`Module '${modulePath}' has no export '${functionName}'`);
  }

  return handlerFn;
}

// ── Configuration from environment ──────────────────────────────────────────────

function _loadConfig() {
  const configStr = process.env.DAGL_CONFIG;
  if (!configStr) {
    throw new Error("DAGL_CONFIG env var not set. Run 'dagl deploy' to configure this function.");
  }
  return JSON.parse(configStr);
}

// ── Initialize at import time (cold start) ──────────────────────────────────────

const userLambda = _importUserHandler();
const config = _loadConfig();

const platform = process.env.FAAS_PLATFORM || 'aws';
const dsType = process.env.UNUM_INTERMEDIARY_DATASTORE_TYPE || 'dynamodb';
const dsName = process.env.UNUM_INTERMEDIARY_DATASTORE_NAME || 'unum-intermediate-datastore';
const gcEnabled = process.env.GC || 'false';

const unum = new Unum(config, dsType, dsName, platform, gcEnabled);

// ── Ingress ─────────────────────────────────────────────────────────────────────

async function ingress(event) {
  if (event.Data.Source === 'http') {
    if (!unum.entryFunction && unum.gc) {
      unum.myGcTasks = event.GC;
    }
    return event.Data.Value;
  }

  // Fan-in: read inputs from datastore
  const eagerFanIn = event.Data.EagerFanIn || {};

  if (eagerFanIn.Enabled) {
    // Eager fan-in: read with await for missing inputs
    const vals = await unum.ds.readInputWithAwait(
      event.Session,
      event.Data.Value,
      parseFloat(process.env.UNUM_EAGER_POLL_INTERVAL || '0.1'),
      parseFloat(process.env.UNUM_EAGER_TIMEOUT || '300')
    );

    if (unum.gc) {
      const gcTasks = {};
      for (const ckpt of vals) {
        if (ckpt.GC) Object.assign(gcTasks, ckpt.GC);
      }
      unum.myGcTasks = gcTasks;
    }
    unum.fanInGc = true;
    return vals.map(ckpt => ckpt.User);
  }

  // Standard fan-in
  const ckptVals = await unum.ds.readInput(event.Session, event.Data.Value);

  if (unum.gc) {
    const gcTasks = {};
    for (const ckpt of ckptVals) {
      if (ckpt.GC) Object.assign(gcTasks, ckpt.GC);
    }
    unum.myGcTasks = gcTasks;
    unum.fanInGc = true;
  }

  return ckptVals.map(ckpt => ckpt.User);
}

// ── Egress ──────────────────────────────────────────────────────────────────────

async function egress(userFunctionOutput, event) {
  // Build checkpoint data
  let checkpointData;
  if (unum.gc) {
    const gc = {
      [unum.getMyInstanceName(event)]: unum.getMyOutgoingEdges(event, userFunctionOutput),
    };
    checkpointData = { GC: gc, User: JSON.stringify(userFunctionOutput) };
  } else {
    checkpointData = { User: JSON.stringify(userFunctionOutput) };
  }

  let session = null;
  let nextPayloadMetadata = null;

  const ret = await unum.runCheckpoint(event, checkpointData);
  if (ret === 0 || ret === -2 || ret === null) {
    [session, nextPayloadMetadata] = await unum.runContinuation(event, userFunctionOutput);
  }

  session = unum.currSession;

  if (unum.gc) {
    await unum.runGc();
  }
  unum.cleanup();

  return [session, nextPayloadMetadata];
}

// ── Handler (Lambda entry point) ────────────────────────────────────────────────

exports.handler = async function handler(event, context) {
  unum.cleanup();

  if (unum.ds && unum.ds.resetMetrics) {
    unum.ds.resetMetrics();
  }

  let inputData = event;

  // Handle non-DAGL events (no envelope)
  if (!inputData.Data) {
    if (unum.entryFunction) {
      // Entry function: auto-wrap into DAGL envelope
      let userPayload, sessionId;

      if (inputData.detail && inputData.source) {
        // EventBridge event
        userPayload = inputData.detail;
        sessionId = inputData.id || randomUUID();
        if (unum.debug) {
          console.log(`[DAGL] EventBridge event from ${inputData.source} / ${inputData['detail-type']}`);
        }
      } else {
        // Raw JSON trigger
        userPayload = inputData;
        sessionId = randomUUID();
        if (unum.debug) {
          console.log(`[DAGL] Raw JSON event, session=${sessionId}`);
        }
      }

      inputData = {
        Data: { Source: 'http', Value: userPayload },
        Session: sessionId,
      };
    } else {
      // Non-entry function invoked directly: passthrough to user handler
      if (unum.debug) {
        console.log('[DAGL] Passthrough: non-DAGL event on non-entry function');
      }
      return userLambda(event, context);
    }
  }

  if (unum.debug) {
    console.log(`[DAGL] Function: ${unum.name}, Session: ${inputData.Session || '?'}`);
  }

  // Check for existing checkpoint (idempotency)
  const ckptRet = await unum.getCheckpoint(inputData);

  let userFunctionOutput;
  if (ckptRet === null) {
    const userFunctionInput = await ingress(inputData);

    if (unum.debug) {
      console.log(`[DAGL] Input: ${JSON.stringify(userFunctionInput)}`);
    }

    // Call user function (support both sync and async handlers)
    userFunctionOutput = await userLambda(userFunctionInput, context);

    if (unum.debug) {
      console.log(`[DAGL] Output: ${JSON.stringify(userFunctionOutput)}`);
    }
  } else {
    userFunctionOutput = ckptRet;
    if (unum.debug) {
      console.log(`[DAGL] Output from checkpoint: ${JSON.stringify(userFunctionOutput)}`);
    }
  }

  await egress(userFunctionOutput, inputData);

  // Log metrics
  if (unum.ds && unum.ds.logMetrics) {
    unum.ds.logMetrics();
  }

  return userFunctionOutput;
};

// ── GCP Cloud Functions v2 HTTP Handler ─────────────────────────────────────────

/**
 * DAGL runtime handler for GCP Cloud Functions v2 (HTTP trigger).
 *
 * Accepts POST requests with JSON body:
 *   - DAGL envelope: {"Data": {...}, "Session": "..."}
 *   - Raw JSON input (entry function): {"text": "..."}
 *
 * Set GCP entry point to: gcpHttpHandler
 *
 * @param {object} req - Express-like request object
 * @param {object} res - Express-like response object
 */
exports.gcpHttpHandler = async function gcpHttpHandler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).send('Method not allowed');
    return;
  }

  const event = req.body;
  if (!event || Object.keys(event).length === 0) {
    res.status(400).json({ error: 'Empty request body' });
    return;
  }

  // ── Core orchestration (same as Lambda handler) ──
  unum.cleanup();

  if (unum.ds && unum.ds.resetMetrics) {
    unum.ds.resetMetrics();
  }

  let inputData;

  if (event.Data) {
    // Already has DAGL envelope
    inputData = event;
  } else if (unum.entryFunction) {
    // Entry function: wrap raw input
    inputData = {
      Data: { Source: 'http', Value: event },
      Session: randomUUID(),
    };
  } else {
    // Non-entry function, no envelope — passthrough
    const result = await userLambda(event, null);
    res.status(200).json(result);
    return;
  }

  if (unum.debug) {
    console.log(`[DAGL GCP] Function: ${unum.name}, Session: ${inputData.Session || '?'}`);
  }

  // Check checkpoint (idempotency)
  const ckptRet = await unum.getCheckpoint(inputData);

  let userFunctionOutput;
  if (ckptRet === null) {
    const userFunctionInput = await ingress(inputData);
    userFunctionOutput = await userLambda(userFunctionInput, null);
  } else {
    userFunctionOutput = ckptRet;
  }

  await egress(userFunctionOutput, inputData);

  if (unum.ds && unum.ds.logMetrics) {
    unum.ds.logMetrics();
  }

  // Return 200 with output (cross-platform callers may read it)
  res.status(200).json(userFunctionOutput);
};
