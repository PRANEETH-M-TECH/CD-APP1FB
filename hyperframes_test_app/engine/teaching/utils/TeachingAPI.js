const TeachingModel = require('../models/TeachingModel');
const TeachingController = require('../controllers/TeachingController');

/**
 * TeachingAPI.js
 * Developer-facing public API to configure lesson objectives and strategies programmatically.
 */
module.exports = {
  /**
   * Instantiates a new Lesson Teaching plan.
   * @param {object} fields 
   * @returns {TeachingModel}
   */
  createLesson: (fields) => {
    return new TeachingModel(fields);
  },

  /**
   * Loads a TeachingController context.
   * @param {TeachingModel} model 
   * @returns {TeachingController}
   */
  loadController: (model) => {
    return new TeachingController(model);
  },

  /**
   * Serializes a teaching model.
   * @param {TeachingModel} model 
   * @returns {object}
   */
  serialize: (model) => {
    return model.serialize();
  },

  /**
   * Deserializes a teaching model.
   * @param {object} json 
   * @returns {TeachingModel}
   */
  deserialize: (json) => {
    return TeachingModel.deserialize(json);
  }
};
