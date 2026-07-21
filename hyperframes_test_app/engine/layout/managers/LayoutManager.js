const FreeLayoutStrategy = require('../strategies/FreeLayoutStrategy');
const ColumnLayoutStrategy = require('../strategies/ColumnLayoutStrategy');
const RowLayoutStrategy = require('../strategies/RowLayoutStrategy');
const Layout = require('../models/Layout');

const strategies = {};

/**
 * LayoutManager.js
 * Central Layout manager mapping registered strategy objects to layout type strings
 * and executing coordinate arrangement calculation passes over Scene Graph components.
 */
class LayoutManager {
  /**
   * Registers a layout calculation strategy.
   * @param {string} type 
   * @param {LayoutStrategy} strategy 
   */
  static registerStrategy(type, strategy) {
    strategies[type.toUpperCase()] = strategy;
  }

  /**
   * Retrieves layout strategy mapping.
   * @param {string} type 
   * @returns {LayoutStrategy}
   */
  static getStrategy(type) {
    return strategies[type.toUpperCase()] || strategies['FREE'];
  }

  /**
   * Arranges scene component coordinates based on Layout configuration.
   * @param {Scene} scene 
   */
  static layoutScene(scene) {
    if (!scene) return;

    const layout = scene.layout || new Layout();
    const strategy = LayoutManager.getStrategy(layout.layoutType);

    // Parent bounding area context (1280x720 canvas coordinates)
    const parentBounds = { x: 0, y: 0, width: 1280, height: 720 };

    // Extract all root components
    const components = (scene.nodes || []).map(n => n.component).filter(Boolean);

    // Update style coordinate configurations on component models directly
    strategy.calculate(components, parentBounds, layout);
  }
}

// Register default layout algorithms
LayoutManager.registerStrategy('FREE', new FreeLayoutStrategy());
LayoutManager.registerStrategy('COLUMN', new ColumnLayoutStrategy());
LayoutManager.registerStrategy('ROW', new RowLayoutStrategy());

module.exports = LayoutManager;
