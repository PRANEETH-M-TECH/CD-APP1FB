/**
 * PerformanceAPI.js
 * Single public API surface for the Hyperframes performance layer.
 * Provides initialize(), cache access, profiler, scheduler, pool, and benchmarking.
 */
const CacheManager       = require('../cache/CacheManager');
const PipelineScheduler  = require('../scheduler/PipelineScheduler');
const ResourcePool       = require('../pooling/ResourcePool');
const Profiler           = require('../profiler/Profiler');
const BenchmarkRunner    = require('../benchmark/BenchmarkRunner');

let _initialized = false;
let _scheduler = null;

class PerformanceAPI {
  /**
   * Initializes the performance layer.
   * Should be called once at engine startup after HyperframesConfig.load().
   * @param {object} config  HyperframesConfig.get() result
   */
  static initialize(config = {}) {
    if (_initialized) return;

    // Boot all named caches
    CacheManager.initialize();

    // Create default scheduler using renderer worker count from config
    const concurrencyLimit = (config.renderer && config.renderer.workerCount) || 4;
    _scheduler = new PipelineScheduler({ concurrencyLimit });

    _initialized = true;
  }

  /**
   * Returns a named CacheStore.
   * @param {string} name
   * @returns {import('../cache/CacheStore')}
   */
  static getCache(name) {
    return CacheManager.getCache(name);
  }

  /**
   * Returns the default PipelineScheduler instance.
   * @returns {PipelineScheduler}
   */
  static getScheduler() {
    if (!_scheduler) _scheduler = new PipelineScheduler();
    return _scheduler;
  }

  /**
   * Creates a new ResourcePool with the provided factory.
   * @param {object} options
   * @returns {ResourcePool}
   */
  static createPool(options) {
    return new ResourcePool(options);
  }

  /**
   * Returns the Profiler class for stage instrumentation.
   * @returns {typeof Profiler}
   */
  static getProfiler() {
    return Profiler;
  }

  /**
   * Runs a named benchmark scenario.
   * @param {string} scenario  'small' | 'medium' | 'large' | 'asset_heavy' | 'animation_heavy'
   * @returns {object}
   */
  static runBenchmark(scenario) {
    return BenchmarkRunner.run(scenario);
  }

  /**
   * Runs all benchmark scenarios and returns a comparative summary.
   * @returns {object[]}
   */
  static runAllBenchmarks() {
    return BenchmarkRunner.runAll();
  }

  /**
   * Returns aggregate stats across all caches and resource pools.
   * @returns {object}
   */
  static diagnostics() {
    return {
      caches: CacheManager.stats(),
      profilerReport: Profiler.report(),
      initialized: _initialized
    };
  }
}

module.exports = PerformanceAPI;
