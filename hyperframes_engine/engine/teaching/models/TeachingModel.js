/**
 * TeachingModel.js
 * Decoupled, serializable Teaching model representing instructional design parameters,
 * lesson goals, strategy configurations, and priority levels.
 */
class TeachingModel {
  constructor(fields = {}) {
    this.teachingId = fields.teaching_id || `teach_${Math.random().toString(36).substr(2, 9)}`;
    this.lessonGoal = fields.lesson_goal || '';
    this.learningObjective = fields.learning_objective || '';
    this.teachingSteps = fields.teaching_steps || []; // Array of TeachingStep objects
    this.dependencies = fields.dependencies || [];
    this.priority = fields.priority !== undefined ? fields.priority : 1;
    this.estimatedDuration = fields.estimated_duration || 0;
    this.teachingStrategy = fields.teaching_strategy || 'sequential'; // sequential, branching
    this.metadata = fields.metadata || {};
  }

  /**
   * Serializes the TeachingModel instance to a pure JSON object.
   * @returns {object}
   */
  serialize() {
    return {
      teaching_id: this.teachingId,
      lesson_goal: this.lessonGoal,
      learning_objective: this.learningObjective,
      teaching_steps: this.teachingSteps.map(s => s.serialize()),
      dependencies: this.dependencies,
      priority: this.priority,
      estimated_duration: this.estimatedDuration,
      teaching_strategy: this.teachingStrategy,
      metadata: this.metadata
    };
  }

  /**
   * Deserializes a TeachingModel instance from a JSON object.
   * @param {object} json 
   * @returns {TeachingModel}
   */
  static deserialize(json) {
    if (!json) return null;
    const TeachingStep = require('./TeachingStep');
    return new TeachingModel({
      ...json,
      teaching_steps: (json.teaching_steps || []).map(s => TeachingStep.deserialize(s))
    });
  }
}

module.exports = TeachingModel;
