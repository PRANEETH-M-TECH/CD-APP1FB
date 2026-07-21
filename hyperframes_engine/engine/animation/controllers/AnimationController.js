const AnimationTimeline = require('../timeline/AnimationTimeline');

/**
 * AnimationController.js
 * Controller responsible for managing visual animation lifecycles and validating keyframes.
 */
class AnimationController {
  constructor(timeline = new AnimationTimeline()) {
    if (!timeline) {
      throw new Error("[AnimationController Error] AnimationTimeline instance is required.");
    }
    this.timeline = timeline;
  }

  /**
   * Asserts logical validation rules on timeline animations.
   * @returns {boolean}
   */
  validate() {
    (this.timeline.animations || []).forEach((anim) => {
      if (!anim.type) {
        throw new Error(`[Animation Validation Error] Animation '${anim.animationId}' is missing an effect type.`);
      }
      if (!anim.target && anim.type !== 'CUSTOM') {
        throw new Error(`[Animation Validation Error] Animation '${anim.animationId}' of type '${anim.type}' must declare a focus target node.`);
      }
    });
    return true;
  }
}

module.exports = AnimationController;
