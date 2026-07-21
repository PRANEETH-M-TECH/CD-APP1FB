const Animation = require('../models/Animation');

/**
 * AnimationTimeline.js
 * Decoupled timeline container holding sequential, parallel, or nested animations.
 */
class AnimationTimeline {
  constructor() {
    this.animations = [];
  }

  /**
   * Appends an Animation instance to this timeline.
   * @param {Animation} animation 
   */
  addAnimation(animation) {
    if (animation) {
      this.animations.push(animation);
    }
  }

  /**
   * Removes an Animation instance by ID.
   * @param {string} animationId 
   */
  removeAnimation(animationId) {
    this.animations = this.animations.filter(a => a.animationId !== animationId);
  }

  /**
   * Serializes the timeline instance to a pure JSON array.
   * @returns {Array<object>}
   */
  serialize() {
    return this.animations.map(a => a.serialize());
  }

  /**
   * Deserializes an AnimationTimeline from a JSON array.
   * @param {Array<object>} json 
   * @returns {AnimationTimeline}
   */
  static deserialize(json) {
    const timeline = new AnimationTimeline();
    if (Array.isArray(json)) {
      json.forEach(item => {
        timeline.addAnimation(Animation.deserialize(item));
      });
    }
    return timeline;
  }
}

module.exports = AnimationTimeline;
