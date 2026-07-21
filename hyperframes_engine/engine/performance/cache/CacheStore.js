/**
 * CacheStore.js
 * LRU in-memory cache with TTL expiration and version-aware keys.
 * Operates independently of all engine modules.
 */

class CacheStore {
  /**
   * @param {object} options
   * @param {number} options.maxSize   Maximum number of entries before LRU eviction
   * @param {number} options.defaultTtlMs  Default TTL in milliseconds (0 = no expiry)
   * @param {string} options.name      Identifier used in profiling output
   */
  constructor({ maxSize = 256, defaultTtlMs = 0, name = 'unnamed' } = {}) {
    this.name = name;
    this.maxSize = maxSize;
    this.defaultTtlMs = defaultTtlMs;
    /** @type {Map<string, { value: *, expiresAt: number|null, insertedAt: number }>} */
    this._store = new Map();
    this._hits = 0;
    this._misses = 0;
    this._evictions = 0;
  }

  /**
   * Builds a version-scoped cache key.
   * @param {string} key 
   * @param {string|number} version 
   * @returns {string}
   */
  static versionedKey(key, version) {
    return `${key}:v${version}`;
  }

  /**
   * Retrieves a cached value. Returns undefined on miss or expiry.
   * @param {string} key 
   * @returns {*}
   */
  get(key) {
    const entry = this._store.get(key);
    if (!entry) {
      this._misses++;
      return undefined;
    }
    if (entry.expiresAt !== null && Date.now() > entry.expiresAt) {
      this._store.delete(key);
      this._misses++;
      return undefined;
    }
    // LRU: re-insert to mark as recently used
    this._store.delete(key);
    this._store.set(key, entry);
    this._hits++;
    return entry.value;
  }

  /**
   * Stores a value. Evicts the LRU entry if at capacity.
   * @param {string} key 
   * @param {*} value 
   * @param {number} [ttlMs]  Overrides defaultTtlMs for this entry
   */
  set(key, value, ttlMs) {
    if (this._store.has(key)) {
      this._store.delete(key);
    } else if (this._store.size >= this.maxSize) {
      // Evict the oldest (first) entry
      const oldest = this._store.keys().next().value;
      this._store.delete(oldest);
      this._evictions++;
    }
    const effectiveTtl = ttlMs !== undefined ? ttlMs : this.defaultTtlMs;
    this._store.set(key, {
      value,
      expiresAt: effectiveTtl > 0 ? Date.now() + effectiveTtl : null,
      insertedAt: Date.now()
    });
  }

  /**
   * Removes a single entry.
   * @param {string} key 
   */
  invalidate(key) {
    this._store.delete(key);
  }

  /**
   * Removes all entries.
   */
  clear() {
    this._store.clear();
  }

  /**
   * Returns cache statistics.
   * @returns {{ hits: number, misses: number, evictions: number, size: number, hitRatio: number }}
   */
  stats() {
    const total = this._hits + this._misses;
    return {
      name: this.name,
      hits: this._hits,
      misses: this._misses,
      evictions: this._evictions,
      size: this._store.size,
      hitRatio: total > 0 ? parseFloat((this._hits / total).toFixed(4)) : 0
    };
  }
}

module.exports = CacheStore;
