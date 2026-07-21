const PedagogicalStrategy = require('../models/PedagogicalStrategy');
const StrategyRegistry = require('../registry/StrategyRegistry');
const PedagogicalStrategyEngine = require('../engine/PedagogicalStrategyEngine');

/**
 * PedagogyAPI.js
 * Developer-facing public API to select and register custom educational strategies.
 */
module.exports = {
  /**
   * Registers a PedagogicalStrategy.
   * @param {string} id 
   * @param {PedagogicalStrategy} strategy 
   */
  registerStrategy: (id, strategy) => {
    StrategyRegistry.register(id, strategy);
  },

  /**
   * Retrieves a PedagogicalStrategy.
   * @param {string} id 
   * @returns {PedagogicalStrategy}
   */
  selectStrategy: (id) => {
    return StrategyRegistry.getStrategy(id);
  },

  /**
   * Sequentially expands and reinforces teaching steps.
   * @param {Array<object>} steps 
   * @param {PedagogicalStrategy} strategy 
   * @returns {Array<object>}
   */
  processSequence: (steps, strategy) => {
    return PedagogicalStrategyEngine.processSequence(steps, strategy);
  },

  /**
   * Serializes a PedagogicalStrategy.
   * @param {PedagogicalStrategy} obj 
   * @returns {object}
   */
  serialize: (obj) => {
    return obj.serialize();
  },

  /**
   * Deserializes a PedagogicalStrategy.
   * @param {object} json 
   * @returns {PedagogicalStrategy}
   */
  deserialize: (json) => {
    return PedagogicalStrategy.deserialize(json);
  },

  /**
   * Lists all strategy IDs.
   * @returns {Array<string>}
   */
  listStrategies: () => {
    return StrategyRegistry.listStrategies();
  }
};
