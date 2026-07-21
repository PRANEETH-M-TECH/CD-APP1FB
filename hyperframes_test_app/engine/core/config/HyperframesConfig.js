/**
 * HyperframesConfig.js
 * Centralized configuration management for the Hyperframes engine.
 * Supports environment modes, feature flags, and per-module enable/disable.
 */
const defaults = {
  env: 'development',  // 'development' | 'production'
  debug: false,
  logLevel: 'info',    // 'trace' | 'debug' | 'info' | 'warn' | 'error' | 'critical'
  renderer: {
    width: 1280,
    height: 720,
    fps: 30,
    outputFormat: 'mp4'
  },
  engine: {
    cameraEnabled: true,
    layoutEnabled: true,
    animationEnabled: true,
    focusEnabled: true,
    themeEnabled: true,
    teachingEnabled: true,
    pedagogyEnabled: true,
    narrationEnabled: true
  },
  features: {
    storyboardValidation: true,
    metricsCollection: true,
    diagnosticsOnStartup: true,
    faultRecovery: true
  },
  assetBasePath: './public/assets'
};

let _config = { ...defaults };

class HyperframesConfig {
  /**
   * Loads configuration from an optional override object and environment variables.
   * @param {object} overrides 
   * @returns {object}
   */
  static load(overrides = {}) {
    _config = { ...defaults, ...overrides };

    if (process.env.NODE_ENV) {
      _config.env = process.env.NODE_ENV;
    }
    if (process.env.HF_LOG_LEVEL) {
      _config.logLevel = process.env.HF_LOG_LEVEL;
    }
    if (process.env.HF_DEBUG === 'true') {
      _config.debug = true;
      _config.logLevel = 'debug';
    }

    return _config;
  }

  /**
   * Returns the active config snapshot.
   * @returns {object}
   */
  static get() {
    return _config;
  }

  /**
   * Retrieves a single config key, with optional dot-notation (e.g. 'engine.cameraEnabled').
   * @param {string} key 
   * @param {*} fallback 
   * @returns {*}
   */
  static getKey(key, fallback = undefined) {
    const parts = key.split('.');
    let cursor = _config;
    for (const part of parts) {
      if (cursor == null || typeof cursor !== 'object') return fallback;
      cursor = cursor[part];
    }
    return cursor !== undefined ? cursor : fallback;
  }

  /**
   * Returns true if the engine is running in production mode.
   * @returns {boolean}
   */
  static isProduction() {
    return _config.env === 'production';
  }

  /**
   * Returns the active log level string.
   * @returns {string}
   */
  static logLevel() {
    return _config.logLevel || 'info';
  }
}

module.exports = HyperframesConfig;
