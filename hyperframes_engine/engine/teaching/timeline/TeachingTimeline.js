const TeachingStep = require('../models/TeachingStep');

/**
 * TeachingTimeline.js
 * Decoupled timeline sequence manager orchestrating instructional action steps
 * and ordering dependencies.
 */
class TeachingTimeline {
  constructor(steps = []) {
    this.steps = steps; // Array of TeachingStep instances
  }

  /**
   * Serializes the timeline steps.
   * @returns {Array<object>}
   */
  serialize() {
    return this.steps.map(s => s.serialize());
  }

  /**
   * Deserializes a TeachingTimeline.
   * @param {Array<object>} json 
   * @returns {TeachingTimeline}
   */
  static deserialize(json) {
    if (!Array.isArray(json)) return new TeachingTimeline();
    return new TeachingTimeline(json.map(item => TeachingStep.deserialize(item)));
  }
}

module.exports = TeachingTimeline;
