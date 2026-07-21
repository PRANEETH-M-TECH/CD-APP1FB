/**
 * Animation.js
 * Decoupled, serializable Animation model representing transitions and effects.
 * Completely renderer-independent and timeline-agnostic.
 */
class Animation {
  constructor(fields = {}) {
    this.animationId = fields.animation_id || `anim_${Math.random().toString(36).substr(2, 9)}`;
    this.type = fields.type || 'FADE_IN'; // FADE_IN, FADE_OUT, MOVE, SCALE, ROTATE, CUSTOM
    this.target = fields.target || null; // Scene Graph node ID, component ID, or CSS selector
    this.duration = fields.duration !== undefined ? fields.duration : 0.5;
    this.delay = fields.delay !== undefined ? fields.delay : 0;
    this.easing = fields.easing || 'power2.out';
    this.direction = fields.direction || 'normal';
    this.repeat = fields.repeat || 0;
    this.trigger = fields.trigger || 'auto'; // auto, click, event
    this.priority = fields.priority || 0;
    this.metadata = fields.metadata || {};
    this.futureConstraints = fields.future_constraints || {};
  }

  /**
   * Serializes the Animation instance to a pure JSON object.
   * @returns {object}
   */
  serialize() {
    return {
      animation_id: this.animationId,
      type: this.type,
      target: this.target,
      duration: this.duration,
      delay: this.delay,
      easing: this.easing,
      direction: this.direction,
      repeat: this.repeat,
      trigger: this.trigger,
      priority: this.priority,
      metadata: this.metadata,
      future_constraints: this.futureConstraints
    };
  }

  /**
   * Deserializes an Animation instance from a JSON object.
   * @param {object} json 
   * @returns {Animation}
   */
  static deserialize(json) {
    if (!json) return null;
    return new Animation(json);
  }
}

module.exports = Animation;
