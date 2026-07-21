/**
 * LayoutStrategy.js
 * Base abstract strategy class representing layout placement algorithms.
 */
class LayoutStrategy {
  constructor(name) {
    this.name = name;
  }

  /**
   * Calculates component positions and updates styling boundaries.
   * @param {Array<Component>} components 
   * @param {object} parentBounds {x, y, width, height}
   * @param {Layout} layoutConfig
   * @returns {Array<Component>}
   */
  calculate(components, parentBounds, layoutConfig) {
    throw new Error(`[LayoutStrategy Error] calculate() not implemented on ${this.name} strategy.`);
  }
}

module.exports = LayoutStrategy;
