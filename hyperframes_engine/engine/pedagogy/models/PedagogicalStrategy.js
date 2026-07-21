/**
 * PedagogicalStrategy.js
 * Decoupled, serializable model representing educational teaching styles,
 * reinforcement schedules, learning style parameters, and instruction paths.
 */
class PedagogicalStrategy {
  constructor(fields = {}) {
    this.strategyId = fields.strategy_id || `strat_${Math.random().toString(36).substr(2, 9)}`;
    this.strategyName = fields.strategy_name || 'Standard Sequence';
    this.subject = fields.subject || 'general';
    this.difficulty = fields.difficulty || 'intermediate';
    this.learningStyle = fields.learning_style || 'visual';
    this.teachingPattern = fields.teaching_pattern || 'definition_explanation_example';
    this.instructionSequence = fields.instruction_sequence || [];
    this.reinforcementRules = fields.reinforcement_rules || [];
    this.assessmentPoints = fields.assessment_points || [];
    this.metadata = fields.metadata || {};
  }

  /**
   * Serializes the PedagogicalStrategy instance to a pure JSON object.
   * @returns {object}
   */
  serialize() {
    return {
      strategy_id: this.strategyId,
      strategy_name: this.strategyName,
      subject: this.subject,
      difficulty: this.difficulty,
      learning_style: this.learningStyle,
      teaching_pattern: this.teachingPattern,
      instruction_sequence: this.instructionSequence,
      reinforcement_rules: this.reinforcementRules,
      assessment_points: this.assessmentPoints,
      metadata: this.metadata
    };
  }

  /**
   * Deserializes a PedagogicalStrategy instance from JSON.
   * @param {object} json 
   * @returns {PedagogicalStrategy}
   */
  static deserialize(json) {
    if (!json) return null;
    return new PedagogicalStrategy(json);
  }
}

module.exports = PedagogicalStrategy;
