/**
 * DAGL Runtime Layer - DynamoDB Datastore Driver (Node.js)
 *
 * Handles checkpointing, fan-in synchronization, and data exchange
 * via DynamoDB. Port of the Python ds.py DynamoDBDriver.
 */

'use strict';

const {
  DynamoDBClient,
  GetItemCommand,
  PutItemCommand,
  DeleteItemCommand,
  UpdateItemCommand,
  BatchGetItemCommand,
} = require('@aws-sdk/client-dynamodb');
const { marshall, unmarshall } = require('@aws-sdk/util-dynamodb');

class DynamoDBDriver {
  /**
   * @param {string} tableName - DynamoDB table name
   * @param {boolean} debug - Enable debug logging
   */
  constructor(tableName, debug = false) {
    this.myType = 'dynamodb';
    this.name = tableName;
    this.debug = debug;
    this.client = new DynamoDBClient({});

    // Metrics
    this._metricsReads = 0;
    this._metricsWrites = 0;
    this._metricsDeletes = 0;
    this._metricsWcu = 0;
    this._metricsRcu = 0;
  }

  resetMetrics() {
    this._metricsReads = 0;
    this._metricsWrites = 0;
    this._metricsDeletes = 0;
    this._metricsWcu = 0;
    this._metricsRcu = 0;
  }

  logMetrics() {
    console.log(
      `[METRICS] dynamo_reads=${this._metricsReads} ` +
      `dynamo_writes=${this._metricsWrites} ` +
      `dynamo_deletes=${this._metricsDeletes} ` +
      `wcu=${this._metricsWcu.toFixed(1)} rcu=${this._metricsRcu.toFixed(1)}`
    );
  }

  // ── Key naming ──────────────────────────────────────────────────────────

  checkpointName(session, instanceName) {
    return `${session}/${instanceName}-output`;
  }

  gcSyncPointName(session, parentInstanceName) {
    return `${session}/${parentInstanceName}-gc`;
  }

  faninSyncPointName(session, aggInstanceName) {
    return `${session}/${aggInstanceName}-fanin`;
  }

  // ── Checkpoint operations ───────────────────────────────────────────────

  /**
   * Write checkpoint (conditional put - only if not exists).
   * @returns {number} 1=success, -1=already exists
   */
  async checkpoint(session, instanceName, data) {
    return this._createIfNotExist(
      'Name',
      this.checkpointName(session, instanceName),
      data
    );
  }

  /**
   * Read checkpoint value, or null if not exists.
   */
  async getCheckpoint(session, instanceName) {
    const key = this.checkpointName(session, instanceName);
    try {
      const resp = await this.client.send(new GetItemCommand({
        TableName: this.name,
        Key: marshall({ Name: key }),
        ConsistentRead: true,
        ReturnConsumedCapacity: 'TOTAL',
      }));

      this._metricsReads++;
      if (resp.ConsumedCapacity) {
        this._metricsRcu += resp.ConsumedCapacity.CapacityUnits || 0;
      }

      if (!resp.Item) return null;
      const item = unmarshall(resp.Item);

      const value = item.User ?? item.Value ?? null;
      if (value === null) return null;
      if (typeof value === 'string') {
        try { return JSON.parse(value); } catch { return value; }
      }
      return value;
    } catch (e) {
      console.warn(`[WARN] getCheckpoint() error: ${e.message}`);
      throw e;
    }
  }

  /**
   * Get full checkpoint item (User + GC), or null.
   */
  async getCheckpointFull(session, instanceName) {
    const key = this.checkpointName(session, instanceName);
    try {
      const resp = await this.client.send(new GetItemCommand({
        TableName: this.name,
        Key: marshall({ Name: key }),
        ConsistentRead: true,
        ReturnConsumedCapacity: 'TOTAL',
      }));

      this._metricsReads++;
      if (resp.ConsumedCapacity) {
        this._metricsRcu += resp.ConsumedCapacity.CapacityUnits || 0;
      }

      if (!resp.Item) return null;
      const item = unmarshall(resp.Item);
      const result = {};

      if (item.User != null) {
        result.User = typeof item.User === 'string' ? JSON.parse(item.User) : item.User;
      } else if (item.Value != null) {
        result.User = typeof item.Value === 'string' ? JSON.parse(item.Value) : item.Value;
      }
      if (item.GC != null) result.GC = item.GC;
      return result;
    } catch (e) {
      if (this.debug) console.log(`[DEBUG] getCheckpointFull() error: ${e.message}`);
      return null;
    }
  }

  async deleteCheckpoint(session, instanceName) {
    return this._delete('Name', this.checkpointName(session, instanceName));
  }

  // ── Fan-in input reading ───────────────────────────────────────────────

  /**
   * Read all fan-in inputs from the datastore.
   * @param {string} session
   * @param {string[]} values - Instance names
   * @returns {Array<{User: any, Name: string, GC?: any}>}
   */
  async readInput(session, values) {
    const itemNames = values.map(v => `${session}/${v}-output`);
    const allRet = [];

    // DynamoDB BatchGetItem limit: 100 keys per request
    for (let i = 0; i < itemNames.length; i += 100) {
      const batch = itemNames.slice(i, i + 100);
      const requestKeys = batch.map(n => marshall({ Name: n }));

      const resp = await this.client.send(new BatchGetItemCommand({
        RequestItems: {
          [this.name]: {
            Keys: requestKeys,
            ConsistentRead: true,
          },
        },
        ReturnConsumedCapacity: 'TOTAL',
      }));

      this._metricsReads += batch.length;
      if (resp.ConsumedCapacity) {
        for (const cap of resp.ConsumedCapacity) {
          this._metricsRcu += cap.CapacityUnits || 0;
        }
      }

      const items = (resp.Responses || {})[this.name] || [];
      for (const raw of items) {
        allRet.push(unmarshall(raw));
      }
    }

    // Sort by original order
    const order = {};
    itemNames.forEach((n, i) => { order[n] = i; });
    allRet.sort((a, b) => order[a.Name] - order[b.Name]);

    const vals = allRet.map(e => {
      const item = {
        User: typeof e.User === 'string' ? JSON.parse(e.User) : (e.User ?? (typeof e.Value === 'string' ? JSON.parse(e.Value) : e.Value)),
        Name: e.Name,
      };
      if (e.GC != null) item.GC = e.GC;
      return item;
    });

    if (vals.length < values.length) {
      console.warn(`[WARN] Not all values for fan-in were read. Expect ${values.length}. Got ${vals.length}`);
    }
    return vals;
  }

  // ── Eager fan-in support ──────────────────────────────────────────────

  /**
   * Atomically claim the right to invoke the fan-in function.
   * @returns {boolean} true if claimed (first caller wins)
   */
  async tryClaimEagerFanin(session, aggInstanceName) {
    const claimName = `${session}/${aggInstanceName}-eager-claim`;
    try {
      await this.client.send(new PutItemCommand({
        TableName: this.name,
        Item: marshall({
          Name: claimName,
          ClaimedAt: new Date().toISOString(),
          Claimed: true,
        }),
        ConditionExpression: 'attribute_not_exists(#N)',
        ExpressionAttributeNames: { '#N': 'Name' },
        ReturnConsumedCapacity: 'TOTAL',
      }));
      this._metricsWrites++;
      return true;
    } catch (e) {
      if (e.name === 'ConditionalCheckFailedException') {
        this._metricsWrites++;
        return false;
      }
      throw e;
    }
  }

  /**
   * Check which checkpoints exist.
   * @returns {[string[], string[]]} [ready, missing] instance names
   */
  async checkCheckpointsExist(session, instanceNames) {
    const itemNames = instanceNames.map(v => `${session}/${v}-output`);
    const ready = [];
    const missing = [];

    for (let i = 0; i < itemNames.length; i += 100) {
      const batch = itemNames.slice(i, i + 100);
      const batchNames = [...batch];
      const requestKeys = batch.map(n => marshall({ Name: n }));

      try {
        const resp = await this.client.send(new BatchGetItemCommand({
          RequestItems: {
            [this.name]: {
              Keys: requestKeys,
              ConsistentRead: true,
              ProjectionExpression: '#N',
              ExpressionAttributeNames: { '#N': 'Name' },
            },
          },
          ReturnConsumedCapacity: 'TOTAL',
        }));

        this._metricsReads += batch.length;
        if (resp.ConsumedCapacity) {
          for (const cap of resp.ConsumedCapacity) {
            this._metricsRcu += cap.CapacityUnits || 0;
          }
        }

        const foundItems = (resp.Responses || {})[this.name] || [];
        const foundNames = new Set(foundItems.map(it => unmarshall(it).Name));

        for (const name of batchNames) {
          const instName = name.replace(`${session}/`, '').replace('-output', '');
          if (foundNames.has(name)) {
            ready.push(instName);
          } else {
            missing.push(instName);
          }
        }
      } catch (e) {
        if (this.debug) console.log(`[DEBUG] Error checking checkpoints: ${e.message}`);
        for (const name of batchNames) {
          const instName = name.replace(`${session}/`, '').replace('-output', '');
          missing.push(instName);
        }
      }
    }

    return [ready, missing];
  }

  /**
   * Poll until all checkpoints exist.
   */
  async awaitCheckpoints(session, instanceNames, pollInterval = 0.1, timeout = 300) {
    const startTime = Date.now();

    while (true) {
      const [ready, missingList] = await this.checkCheckpointsExist(session, instanceNames);
      if (missingList.length === 0) {
        if (this.debug) {
          console.log(`[DEBUG] All ${instanceNames.length} checkpoints ready after ${((Date.now() - startTime) / 1000).toFixed(2)}s`);
        }
        return true;
      }

      const elapsed = (Date.now() - startTime) / 1000;
      if (elapsed >= timeout) {
        throw new Error(`Timeout waiting for checkpoints after ${elapsed.toFixed(2)}s. Missing: ${missingList.join(', ')}`);
      }

      if (this.debug) {
        const preview = missingList.slice(0, 3).join(', ');
        console.log(`[DEBUG] Waiting for ${missingList.length} checkpoints: ${preview}${missingList.length > 3 ? '...' : ''}`);
      }

      await sleep(pollInterval * 1000);
    }
  }

  /**
   * Read inputs with await for missing ones (eager fan-in).
   */
  async readInputWithAwait(session, values, pollInterval = 0.1, timeout = 300) {
    await this.awaitCheckpoints(session, values, pollInterval, timeout);
    return this.readInput(session, values);
  }

  // ── Bitmap-based synchronization ──────────────────────────────────────

  async faninSyncReady(session, aggInstanceName, index, myInstanceName, numBranches) {
    return this._syncReady(this.faninSyncPointName(session, aggInstanceName), index, numBranches);
  }

  async gcSyncReady(session, parentInstanceName, index, numBranches) {
    return this._syncReady(this.gcSyncPointName(session, parentInstanceName), index, numBranches);
  }

  async _syncReady(syncPointName, index, numBranches) {
    await this._createBitmap(syncPointName, numBranches);
    const readyMap = await this._updateBitmapResult(syncPointName, index);
    return this._bitmapReady(readyMap);
  }

  async _createBitmap(bitmapName, bitmapSize) {
    const readyMap = new Array(bitmapSize).fill(false);
    return this._createIfNotExist('Name', bitmapName, { ReadyMap: readyMap });
  }

  async _updateBitmapResult(bitmapName, index) {
    const resp = await this.client.send(new UpdateItemCommand({
      TableName: this.name,
      Key: marshall({ Name: bitmapName }),
      ReturnValues: 'ALL_NEW',
      UpdateExpression: `SET #L[${index}] = :nd`,
      ConditionExpression: 'attribute_exists(#N)',
      ExpressionAttributeValues: marshall({ ':nd': true }),
      ExpressionAttributeNames: { '#N': 'Name', '#L': 'ReadyMap' },
      ReturnConsumedCapacity: 'TOTAL',
    }));

    this._metricsWrites++;
    if (resp.ConsumedCapacity) {
      this._metricsWcu += resp.ConsumedCapacity.CapacityUnits || 0;
    }

    return unmarshall(resp.Attributes).ReadyMap;
  }

  _bitmapReady(bitmap) {
    return bitmap.every(b => b === true);
  }

  // ── Atomic counter for fan-in ─────────────────────────────────────────

  async checkFanInComplete(session, values, targetCount) {
    const counterName = values.join('-') + '-counter';
    const count = await this._updateFanInCounter(session, counterName);
    return count === targetCount;
  }

  async _updateFanInCounter(session, counterName) {
    const key = `${session}/${counterName}`;

    // Create with Count=0 if not exists
    try {
      await this.client.send(new PutItemCommand({
        TableName: this.name,
        Item: marshall({ Name: key, Count: 0 }),
        ConditionExpression: 'attribute_not_exists(#N)',
        ExpressionAttributeNames: { '#N': 'Name' },
        ReturnConsumedCapacity: 'TOTAL',
      }));
      this._metricsWrites++;
    } catch (e) {
      if (e.name === 'ConditionalCheckFailedException') {
        this._metricsWrites++;
      } else {
        throw e;
      }
    }

    // Atomic increment
    const resp = await this.client.send(new UpdateItemCommand({
      TableName: this.name,
      Key: marshall({ Name: key }),
      ReturnValues: 'UPDATED_NEW',
      UpdateExpression: 'SET #C = #C + :incr',
      ConditionExpression: 'attribute_exists(#N)',
      ExpressionAttributeValues: marshall({ ':incr': 1 }),
      ExpressionAttributeNames: { '#N': 'Name', '#C': 'Count' },
      ReturnConsumedCapacity: 'TOTAL',
    }));

    this._metricsWrites++;
    if (resp.ConsumedCapacity) {
      this._metricsWcu += resp.ConsumedCapacity.CapacityUnits || 0;
    }

    return unmarshall(resp.Attributes).Count;
  }

  // ── Internal helpers ──────────────────────────────────────────────────

  async _createIfNotExist(keyName, keyValue, data) {
    const item = { [keyName]: keyValue, ...data };
    try {
      const resp = await this.client.send(new PutItemCommand({
        TableName: this.name,
        Item: marshall(item, { removeUndefinedValues: true }),
        ConditionExpression: 'attribute_not_exists(#N)',
        ExpressionAttributeNames: { '#N': keyName },
        ReturnConsumedCapacity: 'TOTAL',
      }));
      this._metricsWrites++;
      if (resp.ConsumedCapacity) {
        this._metricsWcu += resp.ConsumedCapacity.CapacityUnits || 0;
      }
      return 1;
    } catch (e) {
      if (e.name === 'ConditionalCheckFailedException') {
        return -1;
      }
      throw e;
    }
  }

  async _delete(keyName, keyValue) {
    try {
      const resp = await this.client.send(new DeleteItemCommand({
        TableName: this.name,
        Key: marshall({ [keyName]: keyValue }),
        ReturnConsumedCapacity: 'TOTAL',
      }));
      this._metricsDeletes++;
      if (resp.ConsumedCapacity) {
        this._metricsWcu += resp.ConsumedCapacity.CapacityUnits || 0;
      }
      return 1;
    } catch (e) {
      throw e;
    }
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ─── Redis Driver (cross-cloud shared datastore) ─────────────────────────────────

/**
 * Redis-backed checkpoint/fan-in datastore for cross-cloud workflows.
 *
 * Key schema:
 *   Checkpoint:  dagl:ckpt:{session}/{instance_name}-output → JSON string
 *   GC sync:     dagl:sync:{session}/{parent}-gc           → bitmap via SETBIT
 *   Fan-in sync: dagl:sync:{session}/{agg_func}-fanin      → bitmap via SETBIT
 *   Eager claim: dagl:claim:{session}/{agg_func}-eager      → "1"
 *   Counter:     dagl:ctr:{session}/{counter_name}         → integer
 *
 * Uses the built-in 'net' module to implement a minimal Redis client (RESP protocol).
 * Zero external dependencies.
 */
class RedisDriver {
  /**
   * @param {string} redisUrl - Redis URL e.g. "redis://host:6379/0"
   * @param {boolean} debug - Enable debug logging
   */
  constructor(redisUrl, debug = false) {
    this.myType = 'redis';
    this.name = redisUrl;
    this.debug = debug;

    // Parse Redis URL
    const parsed = new URL(redisUrl);
    this.host = parsed.hostname || '127.0.0.1';
    this.port = parseInt(parsed.port || '6379', 10);
    this.db = parseInt((parsed.pathname || '/0').slice(1) || '0', 10);
    this.password = parsed.password || null;
    this.username = parsed.username || null;
    this.useTls = parsed.protocol === 'rediss:';

    // Connection (lazy)
    this._client = null;
    this._connected = false;

    // Metrics
    this._metricsReads = 0;
    this._metricsWrites = 0;
    this._metricsDeletes = 0;
  }

  resetMetrics() {
    this._metricsReads = 0;
    this._metricsWrites = 0;
    this._metricsDeletes = 0;
  }

  logMetrics() {
    console.log(
      `[METRICS] redis_reads=${this._metricsReads} ` +
      `redis_writes=${this._metricsWrites} ` +
      `redis_deletes=${this._metricsDeletes}`
    );
  }

  // ── Connection management ─────────────────────────────────────────────

  async _ensureConnected() {
    if (this._connected && this._client) return;

    this._buffer = '';
    this._resolveQueue = [];

    if (this.useTls) {
      const tls = require('tls');
      this._client = tls.connect({ host: this.host, port: this.port, rejectUnauthorized: true });
    } else {
      const net = require('net');
      this._client = new net.Socket();
    }

    this._client.on('data', (data) => {
      this._buffer += data.toString();
      this._processBuffer();
    });

    this._client.on('error', (err) => {
      if (this.debug) console.log(`[DAGL Redis] Connection error: ${err.message}`);
      this._connected = false;
    });

    await new Promise((resolve, reject) => {
      if (this.useTls) {
        this._client.on('secureConnect', () => {
          this._connected = true;
          resolve();
        });
        this._client.on('error', reject);
      } else {
        this._client.connect(this.port, this.host, () => {
          this._connected = true;
          resolve();
        });
        this._client.on('error', reject);
      }
    });

    // AUTH if password (supports ACL: AUTH username password)
    if (this.password) {
      if (this.username && this.username !== 'default') {
        await this._command(['AUTH', this.username, this.password]);
      } else {
        await this._command(['AUTH', this.password]);
      }
    }

    // SELECT db
    if (this.db !== 0) {
      await this._command(['SELECT', String(this.db)]);
    }
  }

  _processBuffer() {
    while (this._resolveQueue.length > 0 && this._buffer.length > 0) {
      const result = this._parseResp();
      if (result === undefined) break; // Incomplete data
      const resolver = this._resolveQueue.shift();
      resolver(result);
    }
  }

  _parseResp() {
    if (this._buffer.length === 0) return undefined;

    const type = this._buffer[0];
    const lineEnd = this._buffer.indexOf('\r\n');
    if (lineEnd === -1) return undefined;

    const line = this._buffer.substring(1, lineEnd);

    switch (type) {
      case '+': // Simple string
        this._buffer = this._buffer.substring(lineEnd + 2);
        return line;

      case '-': // Error
        this._buffer = this._buffer.substring(lineEnd + 2);
        return new Error(line);

      case ':': // Integer
        this._buffer = this._buffer.substring(lineEnd + 2);
        return parseInt(line, 10);

      case '$': { // Bulk string
        const len = parseInt(line, 10);
        if (len === -1) {
          this._buffer = this._buffer.substring(lineEnd + 2);
          return null;
        }
        const start = lineEnd + 2;
        const end = start + len;
        if (this._buffer.length < end + 2) return undefined; // Incomplete
        const value = this._buffer.substring(start, end);
        this._buffer = this._buffer.substring(end + 2);
        return value;
      }

      case '*': { // Array
        const count = parseInt(line, 10);
        if (count === -1) {
          this._buffer = this._buffer.substring(lineEnd + 2);
          return null;
        }
        this._buffer = this._buffer.substring(lineEnd + 2);
        const arr = [];
        for (let i = 0; i < count; i++) {
          const elem = this._parseResp();
          if (elem === undefined) {
            // Incomplete — restore buffer (this is simplistic but works for our use case)
            return undefined;
          }
          arr.push(elem);
        }
        return arr;
      }

      default:
        // Unknown type — skip line
        this._buffer = this._buffer.substring(lineEnd + 2);
        return null;
    }
  }

  async _command(args) {
    await this._ensureConnected();

    // Build RESP command
    let cmd = `*${args.length}\r\n`;
    for (const arg of args) {
      const s = String(arg);
      cmd += `$${Buffer.byteLength(s)}\r\n${s}\r\n`;
    }

    return new Promise((resolve, reject) => {
      this._resolveQueue.push((result) => {
        if (result instanceof Error) reject(result);
        else resolve(result);
      });
      this._client.write(cmd);
    });
  }

  // ── Key naming ────────────────────────────────────────────────────────

  checkpointName(session, instanceName) {
    return `dagl:ckpt:${session}/${instanceName}-output`;
  }

  gcSyncPointName(session, parentInstanceName) {
    return `dagl:sync:${session}/${parentInstanceName}-gc`;
  }

  faninSyncPointName(session, aggInstanceName) {
    return `dagl:sync:${session}/${aggInstanceName}-fanin`;
  }

  // ── Checkpoint operations ─────────────────────────────────────────────

  async checkpoint(session, instanceName, data) {
    const key = this.checkpointName(session, instanceName);
    const value = JSON.stringify(data);

    // SET NX EX 3600 — write-once with 1h TTL
    const result = await this._command(['SET', key, value, 'NX', 'EX', '3600']);
    this._metricsWrites++;
    return result === 'OK' ? 1 : -1;
  }

  async getCheckpoint(session, instanceName) {
    const key = this.checkpointName(session, instanceName);
    const raw = await this._command(['GET', key]);
    this._metricsReads++;

    if (raw === null) return null;

    const data = JSON.parse(raw);
    if (data && typeof data === 'object' && 'User' in data) {
      const value = data.User;
      if (typeof value === 'string') {
        try { return JSON.parse(value); } catch { return value; }
      }
      return value;
    }
    return data;
  }

  async getCheckpointFull(session, instanceName) {
    const key = this.checkpointName(session, instanceName);
    const raw = await this._command(['GET', key]);
    this._metricsReads++;

    if (raw === null) return null;

    const data = JSON.parse(raw);
    if (data && typeof data === 'object') {
      const result = {};
      if ('User' in data) {
        result.User = typeof data.User === 'string' ? JSON.parse(data.User) : data.User;
      } else if ('Value' in data) {
        result.User = typeof data.Value === 'string' ? JSON.parse(data.Value) : data.Value;
      }
      if ('GC' in data) result.GC = data.GC;
      return result;
    }
    return { User: data };
  }

  async deleteCheckpoint(session, instanceName) {
    const key = this.checkpointName(session, instanceName);
    await this._command(['DEL', key]);
    this._metricsDeletes++;
  }

  // ── Read input (fan-in) ───────────────────────────────────────────────

  async readInput(session, values) {
    const keys = values.map(v => this.checkpointName(session, v));

    // MGET — single round-trip
    const rawValues = await this._command(['MGET', ...keys]);
    this._metricsReads += keys.length;

    const results = [];
    for (let i = 0; i < rawValues.length; i++) {
      const raw = rawValues[i];
      if (raw === null) {
        console.log(`[WARN] Fan-in checkpoint missing: ${keys[i]}`);
        continue;
      }

      const data = JSON.parse(raw);
      const item = { Name: keys[i] };

      if (data && typeof data === 'object') {
        if ('User' in data) {
          item.User = typeof data.User === 'string' ? JSON.parse(data.User) : data.User;
        } else if ('Value' in data) {
          item.User = typeof data.Value === 'string' ? JSON.parse(data.Value) : data.Value;
        }
        if ('GC' in data) item.GC = data.GC;
      } else {
        item.User = data;
      }

      results.push(item);
    }

    if (results.length < values.length) {
      console.log(`[WARN] Not all fan-in values read from redis. Expect ${values.length}, got ${results.length}`);
    }

    return results;
  }

  async readInputWithAwait(session, values, pollInterval = 100, timeout = 300000) {
    await this.awaitCheckpoints(session, values, pollInterval, timeout);
    return this.readInput(session, values);
  }

  // ── Eager fan-in ──────────────────────────────────────────────────────

  async tryClaimEagerFanin(session, aggInstanceName) {
    const key = `dagl:claim:${session}/${aggInstanceName}-eager-claim`;
    const result = await this._command(['SET', key, '1', 'NX', 'EX', '3600']);
    this._metricsWrites++;
    return result === 'OK';
  }

  async checkCheckpointsExist(session, instanceNames) {
    const ready = [];
    const missing = [];

    // Pipeline EXISTS commands
    for (const name of instanceNames) {
      const key = this.checkpointName(session, name);
      const exists = await this._command(['EXISTS', key]);
      this._metricsReads++;
      if (exists) ready.push(name);
      else missing.push(name);
    }

    return [ready, missing];
  }

  async awaitCheckpoints(session, instanceNames, pollInterval = 100, timeout = 300000) {
    const start = Date.now();

    while (true) {
      const [ready, missing] = await this.checkCheckpointsExist(session, instanceNames);
      if (missing.length === 0) return true;

      const elapsed = Date.now() - start;
      if (elapsed >= timeout) {
        throw new Error(`Timeout waiting for checkpoints. Missing: ${missing.join(', ')}`);
      }

      if (this.debug) {
        console.log(`[DEBUG] Waiting for ${missing.length} checkpoints...`);
      }

      await sleep(pollInterval);
    }
  }

  // ── Synchronization (bitmap via SETBIT/BITCOUNT) ──────────────────────

  async gcSyncReady(session, parentInstanceName, index, numBranches) {
    const syncKey = this.gcSyncPointName(session, parentInstanceName);
    return this._syncReadyBitmap(syncKey, index, numBranches);
  }

  async faninSyncReady(session, aggInstanceName, index, myInstanceName, numBranches) {
    const syncKey = this.faninSyncPointName(session, aggInstanceName);
    return this._syncReadyBitmap(syncKey, index, numBranches);
  }

  async _syncReadyBitmap(syncKey, index, numBranches) {
    // SETBIT + BITCOUNT atomically via EVAL
    const script = `
      redis.call('SETBIT', KEYS[1], ARGV[1], 1)
      redis.call('EXPIRE', KEYS[1], 3600)
      return redis.call('BITCOUNT', KEYS[1])
    `;
    const count = await this._command(['EVAL', script, '1', syncKey, String(index)]);
    this._metricsWrites++;
    return count >= numBranches;
  }

  async checkFanInComplete(session, values, targetCount) {
    const counterName = values.join('-') + '-counter';
    const key = `dagl:ctr:${session}/${counterName}`;
    const count = await this._command(['INCR', key]);
    await this._command(['EXPIRE', key, '3600']);
    this._metricsWrites++;
    return count >= targetCount;
  }
}

module.exports = { DynamoDBDriver, RedisDriver };
