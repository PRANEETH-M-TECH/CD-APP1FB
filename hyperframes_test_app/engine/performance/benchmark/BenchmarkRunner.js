/**
 * BenchmarkRunner.js
 * Repeatable benchmark harness for Hyperframes pipeline performance testing.
 * Supports five scenario categories: small, medium, large, asset_heavy, animation_heavy.
 */
const Profiler = require('../profiler/Profiler');

/**
 * Synthetic scenario definitions.
 * Each scenario describes the workload complexity without rendering actual video.
 */
const SCENARIOS = {
  small: {
    sceneCount: 3,
    assetsPerScene: 2,
    animationsPerScene: 4,
    narrationWordsPerScene: 50,
    teachingSteps: 5
  },
  medium: {
    sceneCount: 8,
    assetsPerScene: 4,
    animationsPerScene: 8,
    narrationWordsPerScene: 120,
    teachingSteps: 12
  },
  large: {
    sceneCount: 20,
    assetsPerScene: 6,
    animationsPerScene: 15,
    narrationWordsPerScene: 200,
    teachingSteps: 30
  },
  asset_heavy: {
    sceneCount: 6,
    assetsPerScene: 20,
    animationsPerScene: 5,
    narrationWordsPerScene: 80,
    teachingSteps: 10
  },
  animation_heavy: {
    sceneCount: 6,
    assetsPerScene: 3,
    animationsPerScene: 40,
    narrationWordsPerScene: 80,
    teachingSteps: 10
  }
};

class BenchmarkRunner {
  /**
   * Returns the list of available scenario names.
   * @returns {string[]}
   */
  static scenarios() {
    return Object.keys(SCENARIOS);
  }

  /**
   * Executes a named benchmark scenario and returns a performance report.
   * @param {string} scenarioName  One of: small, medium, large, asset_heavy, animation_heavy
   * @returns {object}  Benchmark report
   */
  static run(scenarioName) {
    const scenario = SCENARIOS[scenarioName];
    if (!scenario) {
      throw new Error(`BenchmarkRunner: unknown scenario '${scenarioName}'. Valid: ${Object.keys(SCENARIOS).join(', ')}`);
    }

    Profiler.reset();
    const wallStart = Date.now();

    // Stage: Storyboard parse simulation
    Profiler.start('bench_storyboard_parse');
    BenchmarkRunner._simulateWork(scenario.sceneCount * 0.5);
    Profiler.stop('bench_storyboard_parse');

    // Stage: Asset resolution simulation
    Profiler.start('bench_asset_resolution');
    BenchmarkRunner._simulateWork(scenario.sceneCount * scenario.assetsPerScene * 0.2);
    Profiler.stop('bench_asset_resolution');

    // Stage: Teaching plan build simulation
    Profiler.start('bench_teaching_plan');
    BenchmarkRunner._simulateWork(scenario.teachingSteps * 0.3);
    Profiler.stop('bench_teaching_plan');

    // Stage: Layout computation simulation
    Profiler.start('bench_layout');
    BenchmarkRunner._simulateWork(scenario.sceneCount * 0.4);
    Profiler.stop('bench_layout');

    // Stage: Animation timeline compilation
    Profiler.start('bench_animation_compile');
    BenchmarkRunner._simulateWork(scenario.sceneCount * scenario.animationsPerScene * 0.1);
    Profiler.stop('bench_animation_compile');

    // Stage: HTML generation simulation
    Profiler.start('bench_html_generation');
    BenchmarkRunner._simulateWork(scenario.sceneCount * scenario.narrationWordsPerScene * 0.002);
    Profiler.stop('bench_html_generation');

    const totalMs = Date.now() - wallStart;
    const profilerReport = Profiler.report();

    return {
      scenario: scenarioName,
      config: scenario,
      totalDurationMs: totalMs,
      stages: profilerReport.stages,
      slowestStage: profilerReport.slowestStage,
      slowestDurationMs: profilerReport.slowestDurationMs,
      completedAt: new Date().toISOString()
    };
  }

  /**
   * Runs all available scenarios and returns a comparative report.
   * @returns {object[]}
   */
  static runAll() {
    return BenchmarkRunner.scenarios().map(s => BenchmarkRunner.run(s));
  }

  /**
   * Synchronous busy-wait to simulate CPU work (milliseconds).
   * Used to produce meaningful profiler readings without real I/O.
   * @param {number} ms 
   */
  static _simulateWork(ms) {
    const start = Date.now();
    while (Date.now() - start < ms) { /* spin */ }
  }
}

module.exports = BenchmarkRunner;
