const PedagogicalStrategy = require('../models/PedagogicalStrategy');

const registeredStrategies = {};

/**
 * StrategyRegistry.js
 * Strategy library storing reusable educational paths and templates.
 */
class StrategyRegistry {
  /**
   * Registers a pedagogical strategy.
   * @param {string} id 
   * @param {PedagogicalStrategy} strategy 
   */
  static register(id, strategy) {
    if (strategy) {
      registeredStrategies[id] = strategy;
    }
  }

  /**
   * Retrieves a pedagogical strategy by ID.
   * @param {string} id 
   * @returns {PedagogicalStrategy}
   */
  static getStrategy(id) {
    return registeredStrategies[id] || registeredStrategies['default'];
  }

  /**
   * Lists all registered strategy keys.
   * @returns {Array<string>}
   */
  static listStrategies() {
    return Object.keys(registeredStrategies);
  }
}

// Register default strategies matching standard patterns
StrategyRegistry.register('default', new PedagogicalStrategy({
  strategy_id: 'default',
  strategy_name: 'Sequential Explanation',
  teaching_pattern: 'definition_explanation_example',
  instruction_sequence: ['INTRODUCE', 'EXPLAIN', 'EXAMPLE', 'SUMMARIZE']
}));

StrategyRegistry.register('problem_solution', new PedagogicalStrategy({
  strategy_id: 'problem_solution',
  strategy_name: 'Problem-Solution-Verification',
  teaching_pattern: 'problem_solution_verification',
  instruction_sequence: ['INTRODUCE', 'EXPLAIN', 'EXAMPLE', 'REVIEW']
}));

StrategyRegistry.register('comparative', new PedagogicalStrategy({
  strategy_id: 'comparative',
  strategy_name: 'Compare and Contrast',
  teaching_pattern: 'compare_contrast',
  instruction_sequence: ['INTRODUCE', 'COMPARE', 'REVIEW']
}));

module.exports = StrategyRegistry;
