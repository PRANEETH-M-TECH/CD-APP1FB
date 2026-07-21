const LearningObjective = require('../models/LearningObjective');
const StoryboardValidator = require('../validators/StoryboardValidator');
const LegacyPlannerAdapter = require('../adapters/LegacyPlannerAdapter');

/**
 * PlannerAPI.js
 * Developer-facing public API to validate storyboards and adapt older schemas.
 */
module.exports = {
  /**
   * Instantiates a new LearningObjective model.
   * @param {object} fields 
   * @returns {LearningObjective}
   */
  createObjective: (fields) => {
    return new LearningObjective(fields);
  },

  /**
   * Validates a storyboard JSON structure.
   * @param {object} json 
   * @returns {object} { isValid: boolean, errors: Array<string> }
   */
  validate: (json) => {
    return StoryboardValidator.validate(json);
  },

  /**
   * Adapts legacy storyboard configurations.
   * @param {object} json 
   * @returns {object}
   */
  adaptLegacy: (json) => {
    return LegacyPlannerAdapter.adapt(json);
  },

  /**
   * Serializes a LearningObjective model.
   * @param {LearningObjective} obj 
   * @returns {object}
   */
  serialize: (obj) => {
    return obj.serialize();
  },

  /**
   * Deserializes a LearningObjective model.
   * @param {object} json 
   * @returns {LearningObjective}
   */
  deserialize: (json) => {
    return LearningObjective.deserialize(json);
  }
};
