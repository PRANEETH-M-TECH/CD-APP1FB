const LayoutStrategy = require('./LayoutStrategy');

/**
 * FreeLayoutStrategy.js
 * Default positioning strategy (Absolute placement). Leaves all coordinates unchanged,
 * maintaining 100% backward-compatibility for existing visual templates.
 */
class FreeLayoutStrategy extends LayoutStrategy {
  constructor() {
    super('FREE');
  }

  calculate(components, parentBounds, layoutConfig) {
    // Free positioning: leaves coordinate properties untouched
    return components;
  }
}

module.exports = FreeLayoutStrategy;
