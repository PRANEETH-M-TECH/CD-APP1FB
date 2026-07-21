/**
 * TeachingStep.js
 * Represents a single visual instruction action or instructional explain event.
 * Targets Scene Graph entities (nodes, components, groups) rather than HTML elements.
 */
class TeachingStep {
  constructor(fields = {}) {
    this.stepId = fields.step_id || `step_${Math.random().toString(36).substr(2, 9)}`;
    this.actionType = fields.action_type || 'INTRODUCE'; // INTRODUCE, EXPLAIN, EXAMPLE, HIGHLIGHT, COMPARE, SUMMARIZE, REVIEW, TRANSITION
    this.targetId = fields.target_id || null; // Scene node ID or component ID focus target
    this.duration = fields.duration || 5;
    this.script = fields.script || ''; // Narration script associated with step
    this.metadata = fields.metadata || {};
  }

  /**
   * Serializes the TeachingStep instance.
   * @returns {object}
   */
  serialize() {
    return {
      step_id: this.stepId,
      action_type: this.actionType,
      target_id: this.targetId,
      duration: this.duration,
      script: this.script,
      metadata: this.metadata
    };
  }

  /**
   * Deserializes a TeachingStep instance.
   * @param {object} json 
   * @returns {TeachingStep}
   */
  static deserialize(json) {
    if (!json) return null;
    return new TeachingStep(json);
  }
}

module.exports = TeachingStep;
