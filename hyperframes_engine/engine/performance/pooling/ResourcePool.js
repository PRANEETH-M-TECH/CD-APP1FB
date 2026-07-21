/**
 * ResourcePool.js
 * Generic pool of reusable resource instances.
 * Prevents repeated allocation of expensive resources (parsers, serializers, buffers).
 */

class ResourcePool {
  /**
   * @param {object} options
   * @param {function(): *}  options.factory    Creates a new resource instance
   * @param {number}         options.minSize    Pre-allocated instance count (default 1)
   * @param {number}         options.maxSize    Maximum pool size (default 8)
   * @param {string}         options.name       Pool identifier for profiling
   */
  constructor({ factory, minSize = 1, maxSize = 8, name = 'pool' } = {}) {
    if (typeof factory !== 'function') throw new Error('ResourcePool: factory must be a function');
    this.factory = factory;
    this.minSize = minSize;
    this.maxSize = maxSize;
    this.name = name;

    /** @type {Array<*>} Available instances */
    this._available = [];
    /** @type {number} Total created */
    this._totalCreated = 0;
    /** @type {number} Currently in use */
    this._inUse = 0;
    /** @type {number} Total acquisitions */
    this._acquireCount = 0;

    // Pre-allocate minSize instances
    for (let i = 0; i < minSize; i++) {
      this._available.push(this._create());
    }
  }

  _create() {
    this._totalCreated++;
    return this.factory();
  }

  /**
   * Acquires an instance from the pool (or creates one if under maxSize).
   * @returns {*}
   */
  acquire() {
    this._acquireCount++;
    if (this._available.length > 0) {
      const instance = this._available.pop();
      this._inUse++;
      return instance;
    }
    if (this._inUse < this.maxSize) {
      this._inUse++;
      return this._create();
    }
    // Pool exhausted — create a transient instance outside the pool
    return this.factory();
  }

  /**
   * Returns an instance to the pool.
   * @param {*} instance 
   */
  release(instance) {
    if (this._inUse > 0) this._inUse--;
    if (this._available.length < this.maxSize) {
      this._available.push(instance);
    }
    // Otherwise discard (over capacity)
  }

  /**
   * Pool utilization statistics.
   * @returns {object}
   */
  stats() {
    return {
      name: this.name,
      available: this._available.length,
      inUse: this._inUse,
      totalCreated: this._totalCreated,
      acquireCount: this._acquireCount
    };
  }
}

module.exports = ResourcePool;
