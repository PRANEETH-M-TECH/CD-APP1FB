/**
 * HyperframesError.js
 * Unified error model with severity levels, context, and recovery guidance.
 */
const SEVERITY = {
  WARN: 'WARN',
  ERROR: 'ERROR',
  CRITICAL: 'CRITICAL'
};

class HyperframesError extends Error {
  /**
   * @param {string} code          Machine-readable error code (e.g. 'ASSET_NOT_FOUND')
   * @param {string} message       Human-readable description
   * @param {string} severity      WARN | ERROR | CRITICAL
   * @param {object} context       Contextual data about where the error occurred
   * @param {string} recovery      Suggested recovery action
   */
  constructor(code, message, severity = SEVERITY.ERROR, context = {}, recovery = '') {
    super(message);
    this.name = 'HyperframesError';
    this.code = code;
    this.severity = severity;
    this.context = context;
    this.recovery = recovery;
    this.timestamp = new Date().toISOString();
  }

  /**
   * Serializes the error to a structured JSON object.
   * @returns {object}
   */
  toJSON() {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      severity: this.severity,
      context: this.context,
      recovery: this.recovery,
      timestamp: this.timestamp
    };
  }

  /**
   * Factory: validation error.
   * @param {string} message 
   * @param {object} context 
   * @returns {HyperframesError}
   */
  static validation(message, context = {}) {
    return new HyperframesError('VALIDATION_FAILED', message, SEVERITY.WARN, context, 'Review the storyboard JSON schema and fix the reported fields.');
  }

  /**
   * Factory: missing asset error.
   * @param {string} assetId 
   * @param {object} context 
   * @returns {HyperframesError}
   */
  static missingAsset(assetId, context = {}) {
    return new HyperframesError('ASSET_NOT_FOUND', `Asset not found: ${assetId}`, SEVERITY.WARN, context, 'Ensure the asset exists in the configured asset base path.');
  }

  /**
   * Factory: configuration error.
   * @param {string} message 
   * @param {object} context 
   * @returns {HyperframesError}
   */
  static configuration(message, context = {}) {
    return new HyperframesError('CONFIG_INVALID', message, SEVERITY.ERROR, context, 'Review the HyperframesConfig settings.');
  }

  /**
   * Factory: pipeline error.
   * @param {string} stage 
   * @param {string} message 
   * @param {object} context 
   * @returns {HyperframesError}
   */
  static pipeline(stage, message, context = {}) {
    return new HyperframesError('PIPELINE_ERROR', `[${stage}] ${message}`, SEVERITY.ERROR, { stage, ...context }, 'Check the pipeline stage inputs and dependencies.');
  }
}

HyperframesError.SEVERITY = SEVERITY;
module.exports = HyperframesError;
