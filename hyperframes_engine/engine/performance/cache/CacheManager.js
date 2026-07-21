/**
 * CacheManager.js
 * Central registry of named CacheStore instances.
 * Pre-creates all standard Hyperframes engine caches.
 */
const CacheStore = require('./CacheStore');

const CACHE_CONFIGS = {
  storyboard:  { maxSize: 32,  defaultTtlMs: 5 * 60 * 1000 },  // 5 min
  teaching:    { maxSize: 64,  defaultTtlMs: 5 * 60 * 1000 },
  sceneGraph:  { maxSize: 64,  defaultTtlMs: 5 * 60 * 1000 },
  asset:       { maxSize: 512, defaultTtlMs: 30 * 60 * 1000 }, // 30 min
  theme:       { maxSize: 16,  defaultTtlMs: 60 * 60 * 1000 }, // 60 min
  layout:      { maxSize: 64,  defaultTtlMs: 5 * 60 * 1000 },
  render:      { maxSize: 16,  defaultTtlMs: 2 * 60 * 1000 },  // 2 min
  config:      { maxSize: 8,   defaultTtlMs: 60 * 60 * 1000 }
};

/** @type {Map<string, CacheStore>} */
const _registry = new Map();

class CacheManager {
  /**
   * Initializes all standard named caches.
   * Should be called once during engine startup.
   */
  static initialize() {
    for (const [name, opts] of Object.entries(CACHE_CONFIGS)) {
      _registry.set(name, new CacheStore({ name, ...opts }));
    }
  }

  /**
   * Returns a named CacheStore, creating a default one if not registered.
   * @param {string} name 
   * @returns {CacheStore}
   */
  static getCache(name) {
    if (!_registry.has(name)) {
      _registry.set(name, new CacheStore({ name, maxSize: 64 }));
    }
    return _registry.get(name);
  }

  /**
   * Registers a custom CacheStore under a name.
   * @param {string} name 
   * @param {CacheStore} store 
   */
  static register(name, store) {
    _registry.set(name, store);
  }

  /**
   * Clears a single named cache.
   * @param {string} name 
   */
  static invalidate(name) {
    const store = _registry.get(name);
    if (store) store.clear();
  }

  /**
   * Clears all registered caches.
   */
  static invalidateAll() {
    for (const store of _registry.values()) store.clear();
  }

  /**
   * Returns stats for all registered caches.
   * @returns {object[]}
   */
  static stats() {
    return Array.from(_registry.values()).map(s => s.stats());
  }
}

module.exports = CacheManager;
