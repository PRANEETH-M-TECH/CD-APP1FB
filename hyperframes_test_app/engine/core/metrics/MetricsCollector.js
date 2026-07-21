/**
 * MetricsCollector.js
 * Lightweight accumulator for pipeline stage durations and counters.
 * Emits metrics through a well-defined interface.
 */
const _metrics = {};
const _timers = {};

class MetricsCollector {
  /**
   * Starts a named timer.
   * @param {string} stage 
   */
  static start(stage) {
    _timers[stage] = Date.now();
  }

  /**
   * Stops a named timer and records its duration.
   * @param {string} stage 
   * @returns {number} duration in ms
   */
  static stop(stage) {
    const start = _timers[stage];
    if (start === undefined) return 0;
    const durationMs = Date.now() - start;
    _metrics[`${stage}_duration_ms`] = durationMs;
    delete _timers[stage];
    return durationMs;
  }

  /**
   * Increments a named counter.
   * @param {string} key 
   * @param {number} amount 
   */
  static increment(key, amount = 1) {
    _metrics[key] = (_metrics[key] || 0) + amount;
  }

  /**
   * Records a specific metric value.
   * @param {string} key 
   * @param {*} value 
   */
  static record(key, value) {
    _metrics[key] = value;
  }

  /**
   * Returns a snapshot of all collected metrics.
   * @returns {object}
   */
  static summary() {
    return { ...this._metrics, snapshot_at: new Date().toISOString() };
  }

  /**
   * Resets all metrics.
   */
  static reset() {
    Object.keys(_metrics).forEach(k => delete _metrics[k]);
    Object.keys(_timers).forEach(k => delete _timers[k]);
  }

  /**
   * Returns a snapshot of current metrics (static getter workaround).
   */
  static get _metrics() {
    return _metrics;
  }
}

module.exports = MetricsCollector;
