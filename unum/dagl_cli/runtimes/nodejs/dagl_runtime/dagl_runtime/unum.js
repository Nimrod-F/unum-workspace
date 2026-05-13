/**
 * DAGL Runtime Layer - Orchestration Engine (Node.js)
 *
 * Port of the Python unum.py. Handles continuations (scalar, map, fan-in),
 * checkpointing, session management, garbage collection, and instance naming.
 */

'use strict';

const { DynamoDBDriver } = require('./ds');
const { InvocationBackend } = require('./faas_invoke_backend');
const { createBackend, loadFunctionMap, isMultiPlatformMap } = require('./faas_invoke_backend');
const { randomUUID } = require('crypto');

// ── Continuation input types ────────────────────────────────────────────────────

const InputType = {
  SCALAR: 1,
  MAP: 2,
  FAN_IN: 3,
};

// ── Unum class ──────────────────────────────────────────────────────────────────

class Unum {
  /**
   * @param {object} config - unum_config JSON
   * @param {string} datastoreType - 'dynamodb'
   * @param {string} datastoreName - Table name
   * @param {string} platform - 'aws'
   * @param {string|boolean} gc - Garbage collection enabled
   */
  constructor(config, datastoreType, datastoreName, platform, gc) {
    this.name = config.Name;
    this.platform = platform;

    // Parse GC flag
    if (typeof gc === 'string') {
      this.gc = gc.toLowerCase() === 'true';
    } else {
      this.gc = !!gc;
    }

    this.checkpoint = config.Checkpoint || false;
    this.debug = config.Debug || false;
    this.entryFunction = config.Start || false;
    this.nextPayloadModifiers = config['Next Payload Modifiers'] || [];

    // Datastore
    if (datastoreType === 'dynamodb') {
      this.ds = new DynamoDBDriver(datastoreName, this.debug);
    } else if (datastoreType === 'redis') {
      const { RedisDriver } = require('./ds');
      this.ds = new RedisDriver(datastoreName, this.debug);
    } else {
      throw new Error(`Unsupported datastore type: ${datastoreType}`);
    }

    // Build continuations
    this.contList = [];
    if (config.Next) {
      this.faasBackend = createBackend(platform);

      if (Array.isArray(config.Next)) {
        // Count parallel scalar continuations
        let pc = 0;
        for (const c of config.Next) {
          if (c.InputType === 'Scalar') pc++;
        }

        let pi = 0;
        for (const c of config.Next) {
          if (c.InputType === 'Scalar') {
            this.contList.push(new UnumContinuation(
              this.name, c.Name, c.InputType, c.Conditional || null,
              this.faasBackend, datastoreType, this.ds,
              pi, pc, this.debug
            ));
            pi++;
          } else {
            this.contList.push(new UnumContinuation(
              this.name, c.Name, c.InputType, c.Conditional || null,
              this.faasBackend, datastoreType, this.ds,
              -1, 0, this.debug
            ));
          }
        }
      } else if (typeof config.Next === 'object') {
        this.contList.push(new UnumContinuation(
          this.name, config.Next.Name, config.Next.InputType,
          config.Next.Conditional || null,
          this.faasBackend, datastoreType, this.ds,
          -1, 0, this.debug
        ));
      }
    }

    // Per-invocation state
    this.currSession = null;
    this.currInstanceName = null;
    this.currUnumIndexStr = null;
    this.currUnumIndexList = null;
    this.previousCheckpoint = false;
    this.fanInGc = false;
    this.myGcTasks = null;
    this.myOutgoingEdges = null;
  }

  // ── Session management ──────────────────────────────────────────────────

  getSession(inputPayload) {
    if (this.entryFunction) {
      return this._generateSession(inputPayload);
    }
    return this._extractSession(inputPayload);
  }

  _generateSession(inputPayload) {
    if (this.currSession === null) {
      this.currSession = inputPayload.Session || randomUUID();
    }
    return this.currSession;
  }

  _extractSession(inputPayload) {
    if (this.currSession === null) {
      this.currSession = inputPayload.Session;
    }
    return this.currSession;
  }

  // ── Instance naming ─────────────────────────────────────────────────────

  getMyInstanceName(inputPayload) {
    if (this.currInstanceName === null) {
      this.currInstanceName = Unum.computeInstanceName(this.name, inputPayload);
    }
    return this.currInstanceName;
  }

  getMyUnumIndexStr(inputPayload) {
    if (this.currUnumIndexStr === null) {
      this.currUnumIndexStr = Unum.computeUnumIndexStr(inputPayload);
    }
    return this.currUnumIndexStr;
  }

  getMyUnumIndexList(inputPayload) {
    if (this.currUnumIndexList === null) {
      this.currUnumIndexList = Unum.computeUnumIndexList(inputPayload);
    }
    return this.currUnumIndexList;
  }

  static computeInstanceName(functionName, inputPayload) {
    const idx = Unum.computeUnumIndexStr(inputPayload);
    return idx === null ? functionName : `${functionName}-unumIndex-${idx}`;
  }

  static computeUnumIndexList(inputPayload) {
    if (!inputPayload['Fan-out']) return [];
    return Unum._computeUnumIndexList(inputPayload['Fan-out']);
  }

  static _computeUnumIndexList(fanOut) {
    const list = [parseInt(fanOut.Index, 10)];
    if (fanOut.OuterLoop) {
      return list.concat(Unum._computeUnumIndexList(fanOut.OuterLoop));
    }
    return list;
  }

  static computeUnumIndexStr(inputPayload) {
    const list = Unum.computeUnumIndexList(inputPayload);
    return list.length === 0 ? null : list.join('.');
  }

  // ── Checkpoint ────────────────────────────────────────────────────────

  async getCheckpoint(inputPayload) {
    if (!this.debug) return null;

    const session = this.getSession(inputPayload);
    const instanceName = this.getMyInstanceName(inputPayload);
    const dsRet = await this.ds.getCheckpoint(session, instanceName);

    if (dsRet === null) {
      this.previousCheckpoint = false;
      return null;
    }
    this.previousCheckpoint = true;
    return dsRet;
  }

  async runCheckpoint(inputPayload, data) {
    if (this.previousCheckpoint) return -2;
    if (!this.checkpoint && this.noFanInContinuation()) return null;

    const session = this.getSession(inputPayload);
    const instanceName = this.getMyInstanceName(inputPayload);
    const ret = await this.ds.checkpoint(session, instanceName, data);

    if (ret === -1) {
      if (this.debug) {
        console.log(`[DEBUG] checkpoint already exists: ${session}/${instanceName}`);
      }
      return -1;
    }
    if (ret >= 1) return 0;
    return -3;
  }

  noFanInContinuation() {
    return this.contList.every(c => c.inputType !== InputType.FAN_IN);
  }

  hasOnlyScalarContinuations() {
    if (this.contList.length === 0) return true;
    return this.contList.every(c => c.inputType === InputType.SCALAR);
  }

  // ── Continuation execution ────────────────────────────────────────────

  async runContinuation(inputPayload, userFunctionOutput) {
    const session = this.getSession(inputPayload);
    const nextPayloadMetadata = this.runNextPayloadModifiers(inputPayload);

    const gcInfo = {
      [this.getMyInstanceName(inputPayload)]: this.getMyOutgoingEdges(inputPayload, userFunctionOutput),
    };

    for (const c of this.contList) {
      await c.run(
        userFunctionOutput, session, nextPayloadMetadata,
        inputPayload, this.getMyUnumIndexList(inputPayload),
        { gc: gcInfo, myName: this.name, myCurrInstanceName: this.getMyInstanceName(inputPayload) }
      );
    }

    return [session, nextPayloadMetadata];
  }

  getMyOutgoingEdges(inputPayload, userFunctionOutput) {
    if (this.myOutgoingEdges !== null) return this.myOutgoingEdges;

    const postModifierMetadata = this.runNextPayloadModifiers(inputPayload);
    const edges = [];

    for (const c of this.contList) {
      const nextPayload = c.computePayloadMetadata(inputPayload, userFunctionOutput, postModifierMetadata);
      if (nextPayload === null) continue;
      if (Array.isArray(nextPayload)) {
        for (const e of nextPayload) {
          edges.push(Unum.computeInstanceName(c.functionName, e));
        }
      } else {
        edges.push(Unum.computeInstanceName(c.functionName, nextPayload));
      }
    }

    this.myOutgoingEdges = edges;
    return edges;
  }

  // ── Next Payload Modifiers ────────────────────────────────────────────

  runNextPayloadModifiers(inputPayload) {
    let metadata = { 'Fan-out': inputPayload['Fan-out'] || null };
    // Deep copy to avoid mutating original
    metadata = JSON.parse(JSON.stringify(metadata));

    for (const m of this.nextPayloadModifiers) {
      metadata = this._runNextPayloadModifier(m, metadata);
    }
    return metadata;
  }

  _runNextPayloadModifier(modifier, metadata) {
    if (!metadata['Fan-out']) return metadata;

    if (modifier === 'Pop') {
      if (!metadata['Fan-out'].OuterLoop) {
        metadata['Fan-out'] = null;
        return metadata;
      }
      metadata['Fan-out'] = metadata['Fan-out'].OuterLoop;
      return metadata;
    }

    // Variable substitution for exec-style modifiers
    let execMod = modifier;
    if (execMod.includes('$size')) {
      execMod = execMod.replace(/\$size/g, String(metadata['Fan-out'].Size));
    }
    if (execMod.includes('$0')) {
      execMod = execMod.replace(/\$0/g, String(metadata['Fan-out'].Index));
    }
    if (execMod.includes('$1') && metadata['Fan-out'].OuterLoop) {
      execMod = execMod.replace(/\$1/g, String(metadata['Fan-out'].OuterLoop.Index));
    }

    // Execute the modifier (Python used exec(), we use a limited eval approach)
    // Modifiers are compile-time config, so this is safe
    try {
      // Handle assignment-style modifiers like: metadata["Fan-out"]["Size"] = 10
      if (execMod.includes('metadata')) {
        const fn = new Function('metadata', execMod);
        fn(metadata);
      }
    } catch (e) {
      if (this.debug) console.log(`[DEBUG] Modifier error: ${execMod} - ${e.message}`);
    }

    return metadata;
  }

  // ── Garbage collection ────────────────────────────────────────────────

  async runGc() {
    if (!this.myGcTasks) return;
    if (Object.keys(this.myGcTasks).length === 0) return;
    if (!this.checkpoint && !this.fanInGc) return;

    for (const k of Object.keys(this.myGcTasks)) {
      const children = this.myGcTasks[k];
      if (children.length === 1) {
        if (children[0] === this.currInstanceName) {
          await this.ds.deleteCheckpoint(this.currSession, k);
        }
      } else if (children.length > 1) {
        const myIdx = children.indexOf(this.currInstanceName);
        const allReady = await this.ds.gcSyncReady(this.currSession, k, myIdx, children.length);
        if (allReady) {
          await this.ds.deleteCheckpoint(this.currSession, k);
        }
      }
    }
  }

  // ── Cleanup ───────────────────────────────────────────────────────────

  cleanup() {
    this.currSession = null;
    this.currInstanceName = null;
    this.currUnumIndexStr = null;
    this.currUnumIndexList = null;
    this.previousCheckpoint = false;
    this.myGcTasks = null;
    this.myOutgoingEdges = null;
  }

  // ── Static name expansion (used by fan-in) ────────────────────────────

  static expandName(name, inputPayload) {
    let ret = name;
    if (ret.includes('$0')) ret = ret.replace(/\$0/g, String(inputPayload['Fan-out'].Index));
    if (ret.includes('$1')) ret = ret.replace(/\$1/g, String(inputPayload['Fan-out'].OuterLoop.Index));
    if (ret.includes('$2')) ret = ret.replace(/\$2/g, String(inputPayload['Fan-out'].OuterLoop.OuterLoop.Index));
    if (ret.includes('$3')) ret = ret.replace(/\$3/g, String(inputPayload['Fan-out'].OuterLoop.OuterLoop.OuterLoop.Index));
    return ret;
  }
}

// ── UnumContinuation class ──────────────────────────────────────────────────────

class UnumContinuation {
  /**
   * @param {string} myNodeName - Parent function name
   * @param {string} functionName - Continuation function name
   * @param {string|object} inputType - 'Scalar', 'Map', or {Fan-in: {Values: [...]}}
   * @param {string|null} conditional - Conditional expression
   * @param {InvocationBackend} invoker
   * @param {string} datastoreType
   * @param {DynamoDBDriver} datastore
   * @param {number} parallelIndex
   * @param {number} parallelSize
   * @param {boolean} debug
   */
  constructor(myNodeName, functionName, inputType, conditional, invoker, datastoreType, datastore, parallelIndex = -1, parallelSize = 0, debug = false) {
    this.myNodeName = myNodeName;
    this.functionName = functionName;
    this.invoker = invoker;
    this.conditional = conditional;
    this.datastore = datastore;
    this.debug = debug;

    if (inputType === 'Scalar') {
      this.inputType = InputType.SCALAR;
      this.parallelIndex = parallelIndex;
      this.parallelSize = parallelSize;
    } else if (inputType === 'Map') {
      this.inputType = InputType.MAP;
    } else if (typeof inputType === 'object' && inputType['Fan-in']) {
      this.inputType = InputType.FAN_IN;
      this.fanInValues = inputType['Fan-in'].Values;
    } else {
      throw new Error(`Unknown InputType: ${JSON.stringify(inputType)}`);
    }
  }

  /**
   * Execute this continuation.
   */
  async run(userFunctionOutput, session, nextPayloadMetadata, inputPayload, unumIndexList, kwargs) {
    if (this.inputType === InputType.SCALAR) {
      return this._runScalar(userFunctionOutput, session, nextPayloadMetadata, inputPayload, unumIndexList, kwargs);
    } else if (this.inputType === InputType.MAP) {
      return this._runMap(userFunctionOutput, session, nextPayloadMetadata, inputPayload, unumIndexList, kwargs);
    } else if (this.inputType === InputType.FAN_IN) {
      return this._runFanIn(userFunctionOutput, session, nextPayloadMetadata, inputPayload, unumIndexList, kwargs);
    }
  }

  // ── Conditional check ─────────────────────────────────────────────────

  checkConditional(userFunctionOutput, inputPayload, unumIndexList) {
    if (this.conditional === null) return true;

    let cond = this.conditional;
    if (cond.includes('$size')) cond = cond.replace(/\$size/g, String(inputPayload['Fan-out'].Size));
    if (cond.includes('$0')) cond = cond.replace(/\$0/g, String(unumIndexList[0]));
    if (cond.includes('$1')) cond = cond.replace(/\$1/g, String(unumIndexList[1]));
    if (cond.includes('$2')) cond = cond.replace(/\$2/g, String(unumIndexList[2]));

    if (cond.includes('$ret')) {
      if (typeof userFunctionOutput === 'string') {
        cond = cond.replace(/\$ret/g, `'${userFunctionOutput}'`);
      }
    }

    // Evaluate the conditional (config-time static expression)
    try {
      return !!Function(`"use strict"; return (${cond})`)();
    } catch (e) {
      if (this.debug) console.log(`[DEBUG] Conditional eval error: ${cond} - ${e.message}`);
      return false;
    }
  }

  // ── Payload metadata computation ──────────────────────────────────────

  computePayloadMetadata(inputPayload, userFunctionOutput, postModifierMetadata) {
    if (!this.checkConditional(userFunctionOutput, inputPayload, Unum.computeUnumIndexList(inputPayload))) {
      return null;
    }

    if (this.inputType === InputType.SCALAR) {
      const payload = {};
      if (this.parallelSize > 1) {
        payload['Fan-out'] = {
          Type: 'Parallel',
          Index: this.parallelIndex,
          Size: this.parallelSize,
        };
      }
      for (const f of Object.keys(postModifierMetadata)) {
        if (postModifierMetadata[f] !== null) {
          if (f === 'Fan-out' && payload['Fan-out']) {
            payload['Fan-out'].OuterLoop = postModifierMetadata[f];
          } else {
            payload[f] = postModifierMetadata[f];
          }
        }
      }
      return payload;
    }

    if (this.inputType === InputType.FAN_IN) {
      const payload = {};
      for (const f of Object.keys(postModifierMetadata)) {
        if (postModifierMetadata[f] !== null) {
          if (f === 'Fan-out' && payload['Fan-out']) {
            payload['Fan-out'].OuterLoop = postModifierMetadata[f];
          } else {
            payload[f] = postModifierMetadata[f];
          }
        }
      }
      return payload;
    }

    if (this.inputType === InputType.MAP) {
      const ret = [];
      for (let i = 0; i < userFunctionOutput.length; i++) {
        const payload = {
          'Fan-out': { Type: 'Map', Index: i, Size: userFunctionOutput.length },
        };
        for (const f of Object.keys(postModifierMetadata)) {
          if (postModifierMetadata[f] !== null) {
            if (f === 'Fan-out') {
              payload['Fan-out'].OuterLoop = postModifierMetadata[f];
            } else {
              payload[f] = postModifierMetadata[f];
            }
          }
        }
        ret.push(payload);
      }
      return ret;
    }

    return null;
  }

  // ── Scalar continuation ───────────────────────────────────────────────

  async _runScalar(userFunctionOutput, session, nextPayloadMetadata, inputPayload, unumIndexList, kwargs) {
    if (!this.checkConditional(userFunctionOutput, inputPayload, unumIndexList)) return;

    const payload = {
      Data: { Source: 'http', Value: userFunctionOutput },
      Session: session,
    };

    if (this.parallelSize > 1) {
      payload['Fan-out'] = {
        Type: 'Parallel',
        Index: this.parallelIndex,
        Size: this.parallelSize,
      };
    }

    for (const f of Object.keys(nextPayloadMetadata)) {
      if (nextPayloadMetadata[f] !== null) {
        if (f === 'Fan-out' && payload['Fan-out']) {
          payload['Fan-out'].OuterLoop = nextPayloadMetadata[f];
        } else {
          payload[f] = nextPayloadMetadata[f];
        }
      }
    }

    payload.GC = kwargs.gc;

    if (this.debug) {
      console.log(`[DEBUG] ${this.myNodeName}-${unumIndexList} invoking ${this.functionName} with ${JSON.stringify(payload)}`);
    }

    return this.invoker.invoke(this.functionName, payload);
  }

  // ── Map continuation ──────────────────────────────────────────────────

  async _runMap(userFunctionOutput, session, nextPayloadMetadata, inputPayload, unumIndexList, kwargs) {
    if (!this.checkConditional(userFunctionOutput, inputPayload, unumIndexList)) return;

    if (!Array.isArray(userFunctionOutput)) {
      console.error('[Error] Map continuations expect user function output to be an array.');
      return;
    }

    const size = userFunctionOutput.length;
    for (let i = 0; i < size; i++) {
      const payload = {
        Data: { Source: 'http', Value: userFunctionOutput[i] },
        Session: session,
        'Fan-out': { Type: 'Map', Index: i, Size: size },
      };

      for (const f of Object.keys(nextPayloadMetadata)) {
        if (nextPayloadMetadata[f] !== null) {
          if (f === 'Fan-out') {
            payload['Fan-out'].OuterLoop = nextPayloadMetadata[f];
          } else {
            payload[f] = nextPayloadMetadata[f];
          }
        }
      }

      payload.GC = kwargs.gc;

      if (this.debug) {
        console.log(`[DEBUG] ${this.myNodeName}-${unumIndexList} invoking ${this.functionName} (map ${i}/${size})`);
      }

      await this.invoker.invoke(this.functionName, payload);
    }
  }

  // ── Fan-in continuation ───────────────────────────────────────────────

  async _runFanIn(userFunctionOutput, session, nextPayloadMetadata, inputPayload, unumIndexList, kwargs) {
    const eagerMode = (process.env.EAGER || 'false').toLowerCase();
    const isEager = eagerMode === 'true' || eagerMode === '1' || eagerMode === 'yes';

    if (isEager) {
      return this._runFanInEager(userFunctionOutput, session, nextPayloadMetadata, inputPayload, unumIndexList, kwargs);
    }
    return this._runFanInClassic(userFunctionOutput, session, nextPayloadMetadata, inputPayload, unumIndexList, kwargs);
  }

  async _runFanInClassic(userFunctionOutput, session, nextPayloadMetadata, inputPayload, unumIndexList, kwargs) {
    // Build aggregation function instance name (skip Fan-out)
    const aggPayload = {};
    for (const f of Object.keys(nextPayloadMetadata)) {
      if (nextPayloadMetadata[f] !== null && f !== 'Fan-out') {
        aggPayload[f] = nextPayloadMetadata[f];
      }
    }
    const aggInstanceName = Unum.computeInstanceName(this.functionName, aggPayload);

    // Determine my index in the bitmap
    let myIndex = -1;
    for (let i = 0; i < this.fanInValues.length; i++) {
      if (this.fanInValues[i].startsWith(this.myNodeName)) {
        if (this.fanInValues[i].endsWith('*')) {
          myIndex = unumIndexList[0];
        } else {
          myIndex = i;
        }
      }
    }

    const branchInstanceNames = this._expandAllFanInValueNames(unumIndexList, inputPayload);
    const numBranches = branchInstanceNames.length;

    // Register readiness
    const allReady = await this.datastore.faninSyncReady(session, aggInstanceName, myIndex, kwargs.myCurrInstanceName, numBranches);

    if (allReady) {
      const payload = { ...aggPayload };
      payload.Data = { Source: this.datastore.myType, Value: branchInstanceNames };
      payload.Session = session;

      if (this.debug) {
        console.log(`[DEBUG] ${this.myNodeName}-${unumIndexList} CLASSIC invoking ${this.functionName} (all branches ready)`);
      }

      await this.invoker.invoke(this.functionName, payload);
    }
  }

  async _runFanInEager(userFunctionOutput, session, nextPayloadMetadata, inputPayload, unumIndexList, kwargs) {
    // Build aggregation function instance name (skip Fan-out)
    const aggPayload = {};
    for (const f of Object.keys(nextPayloadMetadata)) {
      if (nextPayloadMetadata[f] !== null && f !== 'Fan-out') {
        aggPayload[f] = nextPayloadMetadata[f];
      }
    }
    const aggInstanceName = Unum.computeInstanceName(this.functionName, aggPayload);

    const branchInstanceNames = this._expandAllFanInValueNames(unumIndexList, inputPayload);
    const numBranches = branchInstanceNames.length;

    // Atomic claim
    const claimed = await this.datastore.tryClaimEagerFanin(session, aggInstanceName);

    if (claimed) {
      const [ready, missingList] = await this.datastore.checkCheckpointsExist(session, branchInstanceNames);

      const payload = { ...aggPayload };
      payload.Data = {
        Source: this.datastore.myType,
        Value: branchInstanceNames,
        EagerFanIn: {
          Enabled: true,
          Ready: ready,
          Missing: missingList,
          TotalBranches: numBranches,
        },
      };
      payload.Session = session;

      if (this.debug) {
        console.log(`[DEBUG] ${this.myNodeName} EAGER invoking ${this.functionName}. Ready: ${ready.length}/${numBranches}`);
      }

      await this.invoker.invoke(this.functionName, payload);
    }

    // Also update sync bitmap
    let myIndex = -1;
    for (let i = 0; i < this.fanInValues.length; i++) {
      if (this.fanInValues[i].startsWith(this.myNodeName)) {
        if (this.fanInValues[i].endsWith('*')) {
          myIndex = unumIndexList[0];
        } else {
          myIndex = i;
        }
      }
    }
    await this.datastore.faninSyncReady(session, aggInstanceName, myIndex, kwargs.myCurrInstanceName, numBranches);
  }

  // ── Fan-in name expansion ─────────────────────────────────────────────

  _expandAllFanInValueNames(unumIndexList, inputPayload) {
    const expanded = [];
    for (const n of this.fanInValues) {
      const tmp = UnumContinuation.expandName(n, inputPayload, unumIndexList, true);
      if (Array.isArray(tmp)) {
        expanded.push(...tmp);
      } else {
        expanded.push(tmp);
      }
    }
    return expanded;
  }

  /**
   * Expand a fan-in value name by replacing $0, $1, (...) expressions, and *.
   */
  static expandName(name, inputPayload, unumIndexList, expandStar = false) {
    let ret = name;

    // Replace positional variables $0, $1, ...
    const positionals = ret.match(/\$\d/g) || [];
    for (const p of positionals) {
      const idx = parseInt(p[1], 10);
      ret = ret.replace(p, String(unumIndexList[idx]));
    }

    // Evaluate parenthesized expressions: (expr)
    const expWithParens = ret.match(/\([^)]+\)/g) || [];
    for (const expr of expWithParens) {
      const inner = expr.slice(1, -1);
      try {
        const val = Function(`"use strict"; return (${inner})`)();
        ret = ret.replace(expr, String(val));
      } catch { /* ignore */ }
    }

    // Expand * wildcard
    if (expandStar && ret.includes('*')) {
      const size = inputPayload['Fan-out'].Size;
      return Array.from({ length: size }, (_, i) => ret.replace('*', String(i)));
    }

    return ret;
  }
}

module.exports = { Unum, UnumContinuation, InputType };
