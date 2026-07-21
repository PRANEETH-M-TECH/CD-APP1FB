/**
 * Logger.js
 * Structured JSON logger with levels and correlation ID support.
 * Replaces ad hoc console.log usage across the engine.
 */
const LEVELS = { trace: 0, debug: 1, info: 2, warn: 3, error: 4, critical: 5 };

let _minLevel = 'info';
let _correlationId = null;

class Logger {
  /**
   * Configures the minimum log level.
   * @param {string} level 
   */
  static setLevel(level) {
    _minLevel = level || 'info';
  }

  /**
   * Sets a correlation ID to be included in all log lines.
   * @param {string} id 
   */
  static setCorrelationId(id) {
    _correlationId = id;
  }

  /**
   * Core log emitter.
   * @param {string} level 
   * @param {string} message 
   * @param {object} meta 
   */
  static log(level, message, meta = {}) {
    if ((LEVELS[level] || 0) < (LEVELS[_minLevel] || 0)) return;
    const line = {
      ts: new Date().toISOString(),
      level,
      ...((_correlationId) ? { correlationId: _correlationId } : {}),
      message,
      ...meta
    };
    const output = JSON.stringify(line);
    if (level === 'error' || level === 'critical') {
      console.error(`[hyperframes] ${output}`);
    } else if (level === 'warn') {
      console.warn(`[hyperframes] ${output}`);
    } else {
      console.log(`[hyperframes] ${output}`);
    }
  }

  static trace(msg, meta = {}) { Logger.log('trace', msg, meta); }
  static debug(msg, meta = {}) { Logger.log('debug', msg, meta); }
  static info(msg, meta = {}) { Logger.log('info', msg, meta); }
  static warn(msg, meta = {}) { Logger.log('warn', msg, meta); }
  static error(msg, meta = {}) { Logger.log('error', msg, meta); }
  static critical(msg, meta = {}) { Logger.log('critical', msg, meta); }
}

module.exports = Logger;
