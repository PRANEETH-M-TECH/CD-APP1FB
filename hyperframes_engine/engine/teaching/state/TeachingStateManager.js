/**
 * TeachingStateManager.js
 * Tracks active learning objectives, completed steps, current index, and active focus concept.
 */
class TeachingStateManager {
  constructor(teachingModel) {
    this.model = teachingModel;
    this.currentStepIndex = 0;
    this.completedSteps = [];
    this.activeConcept = '';
    this.currentObjective = teachingModel ? teachingModel.learningObjective : '';
  }

  /**
   * Returns active TeachingStep configuration.
   * @returns {TeachingStep|null}
   */
  getCurrentStep() {
    if (!this.model) return null;
    return this.model.teachingSteps[this.currentStepIndex] || null;
  }

  /**
   * Advances the instructional flow index.
   * @returns {boolean}
   */
  moveToNext() {
    if (!this.model || this.currentStepIndex >= this.model.teachingSteps.length - 1) {
      return false;
    }
    const current = this.getCurrentStep();
    if (current) {
      this.completedSteps.push(current.stepId);
    }
    this.currentStepIndex++;
    return true;
  }

  /**
   * Recedes the instructional flow index.
   * @returns {boolean}
   */
  moveToPrevious() {
    if (this.currentStepIndex <= 0) {
      return false;
    }
    this.currentStepIndex--;
    this.completedSteps.pop();
    return true;
  }
}

module.exports = TeachingStateManager;
