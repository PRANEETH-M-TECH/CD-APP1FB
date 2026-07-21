const TeachingStateManager = require('../state/TeachingStateManager');

/**
 * TeachingController.js
 * Decoupled controller directing the educational execution pipeline, progress tracking,
 * and next/prev step coordination.
 */
class TeachingController {
  constructor(teachingModel) {
    if (!teachingModel) {
      throw new Error("[TeachingController Error] TeachingModel instance is required.");
    }
    this.model = teachingModel;
    this.stateManager = new TeachingStateManager(teachingModel);
  }

  /**
   * Retrieves active step instruction.
   * @returns {TeachingStep|null}
   */
  getCurrentInstruction() {
    return this.stateManager.getCurrentStep();
  }

  /**
   * Triggers next pacing step.
   * @returns {boolean}
   */
  next() {
    return this.stateManager.moveToNext();
  }

  /**
   * Triggers previous step.
   * @returns {boolean}
   */
  prev() {
    return this.stateManager.moveToPrevious();
  }
}

module.exports = TeachingController;
