/**
 * PipelineScheduler.js
 * Dependency-aware task queue with concurrent slot execution.
 * Executes independent tasks in parallel while respecting declared dependencies.
 * Avoids shared mutable state between task functions.
 */

class PipelineScheduler {
  /**
   * @param {object} options
   * @param {number} options.concurrencyLimit  Max parallel tasks (default 4)
   */
  constructor({ concurrencyLimit = 4 } = {}) {
    this.concurrencyLimit = concurrencyLimit;
  }

  /**
   * Executes a list of task descriptors respecting dependencies.
   *
   * @param {Array<{ name: string, deps: string[], task: function(results: object): Promise<*> }>} descriptors
   * @returns {Promise<{ results: object, timings: object, order: string[] }>}
   */
  async run(descriptors) {
    const results = {};
    const timings = {};
    const order = [];
    const completed = new Set();
    const pending = new Map(descriptors.map(d => [d.name, d]));
    const running = new Map();

    const isReady = (d) => d.deps.every(dep => completed.has(dep));

    const executeTask = async (descriptor) => {
      const start = Date.now();
      try {
        results[descriptor.name] = await descriptor.task(results);
      } catch (err) {
        results[descriptor.name] = { error: err.message };
      }
      timings[descriptor.name] = Date.now() - start;
      order.push(descriptor.name);
      completed.add(descriptor.name);
      running.delete(descriptor.name);
    };

    // Drain loop
    while (pending.size > 0 || running.size > 0) {
      // Launch all ready tasks up to concurrency limit
      for (const [name, descriptor] of pending) {
        if (running.size >= this.concurrencyLimit) break;
        if (isReady(descriptor)) {
          pending.delete(name);
          const promise = executeTask(descriptor);
          running.set(name, promise);
        }
      }

      if (running.size === 0 && pending.size > 0) {
        throw new Error(
          `PipelineScheduler: circular or unsatisfiable dependency detected. Remaining: ${[...pending.keys()].join(', ')}`
        );
      }

      // Wait for the fastest running task to finish before re-evaluating
      if (running.size > 0) {
        await Promise.race([...running.values()]);
      }
    }

    return { results, timings, order };
  }
}

module.exports = PipelineScheduler;
