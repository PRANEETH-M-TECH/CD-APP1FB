/**
 * Profiler.js
 * Lightweight per-stage CPU timing, memory snapshots, and cache-hit tracking.
 * Extends MetricsCollector from the core layer with high-resolution instrumentation.
 */
const MetricsCollector = require('../../core/metrics/MetricsCollector');

/** @type {Map<string, bigint>} hrtime start points */
const _hrStarts = new Map();
/** @type {object[]} captured profiling records */
const _records = [];
/** @type {object} cache hit/miss accumulators */
const _cacheStats = {};

class Profiler {
  /**
   * Starts a high-resolution timer for a named stage.
   * Also captures a baseline memory snapshot.
   * @param {string} stage 
   */
  static start(stage) {
    _hrStarts.set(stage, process.hrtime.bigint());
    MetricsCollector.start(stage);
  }

  /**
   * Stops the high-resolution timer and records the result.
   * @param {string} stage 
   * @returns {{ stage: string, durationMs: number, memoryMb: number }}
   */
  static stop(stage) {
    const hrStart = _hrStarts.get(stage);
    let durationMs = 0;
    if (hrStart !== undefined) {
      durationMs = Number(process.hrtime.bigint() - hrStart) / 1e6;
      _hrStarts.delete(stage);
    }
    MetricsCollector.stop(stage);

    const mem = process.memoryUsage();
    const record = {
      stage,
      durationMs: parseFloat(durationMs.toFixed(3)),
      heapUsedMb: parseFloat((mem.heapUsed / 1024 / 1024).toFixed(2)),
      rssUsedMb:  parseFloat((mem.rss      / 1024 / 1024).toFixed(2)),
      ts: new Date().toISOString()
    };
    _records.push(record);
    return record;
  }

  /**
   * Records a cache event for profiling.
   * @param {string} cacheName 
   * @param {'hit'|'miss'} event 
   */
  static recordCacheEvent(cacheName, event) {
    if (!_cacheStats[cacheName]) _cacheStats[cacheName] = { hits: 0, misses: 0 };
    _cacheStats[cacheName][event === 'hit' ? 'hits' : 'misses']++;
  }

  /**
   * Returns the full profiling report.
   * @returns {{ stages: object[], cacheStats: object, slowest: string|null, metrics: object }}
   */
  static report() {
    const slowest = _records.reduce((s, r) => (!s || r.durationMs > s.durationMs ? r : s), null);
    return {
      stages: _records.slice(),
      cacheStats: { ..._cacheStats },
      slowestStage: slowest ? slowest.stage : null,
      slowestDurationMs: slowest ? slowest.durationMs : 0,
      metrics: MetricsCollector.summary()
    };
  }

  /**
   * Resets all profiler state.
   */
  static reset() {
    _hrStarts.clear();
    _records.length = 0;
    Object.keys(_cacheStats).forEach(k => delete _cacheStats[k]);
    MetricsCollector.reset();
  }
}

module.exports = Profiler;
