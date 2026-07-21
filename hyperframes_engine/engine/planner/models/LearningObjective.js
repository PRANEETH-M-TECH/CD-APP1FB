/**
 * LearningObjective.js
 * Decoupled, serializable model representing educational targets, student level,
 * difficulty thresholds, misconceptions, and assessment markers.
 */
class LearningObjective {
  constructor(fields = {}) {
    this.lessonGoal = fields.lesson_goal || '';
    this.learningObjectives = fields.learning_objectives || [];
    this.studentLevel = fields.student_level || 'General';
    this.estimatedDuration = fields.estimated_duration || 0;
    this.difficulty = fields.difficulty || 'intermediate'; // beginner, intermediate, advanced
    this.subject = fields.subject || 'general';
    this.chapter = fields.chapter || '';
    this.concepts = fields.concepts || [];
    this.prerequisites = fields.prerequisites || [];
    this.misconceptions = fields.misconceptions || [];
    this.assessmentPoints = fields.assessment_points || [];
    this.summary = fields.summary || '';
    this.metadata = fields.metadata || {};
  }

  /**
   * Serializes the LearningObjective model to a pure JSON object.
   * @returns {object}
   */
  serialize() {
    return {
      lesson_goal: this.lessonGoal,
      learning_objectives: this.learningObjectives,
      student_level: this.studentLevel,
      estimated_duration: this.estimatedDuration,
      difficulty: this.difficulty,
      subject: this.subject,
      chapter: this.chapter,
      concepts: this.concepts,
      prerequisites: this.prerequisites,
      misconceptions: this.misconceptions,
      assessment_points: this.assessmentPoints,
      summary: this.summary,
      metadata: this.metadata
    };
  }

  /**
   * Deserializes a LearningObjective model instance from JSON.
   * @param {object} json 
   * @returns {LearningObjective}
   */
  static deserialize(json) {
    if (!json) return null;
    return new LearningObjective(json);
  }
}

module.exports = LearningObjective;
