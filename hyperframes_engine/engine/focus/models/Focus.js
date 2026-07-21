/**
 * Focus.js
 * Decoupled, serializable Focus model representing educational emphasis guidelines.
 * Completely renderer-independent and layout-agnostic.
 */
class Focus {
  constructor(fields = {}) {
    this.focusId = fields.focus_id || `focus_${Math.random().toString(36).substr(2, 9)}`;
    this.target = fields.target || null; // Scene node ID, component ID, or Group ID
    this.priority = fields.priority !== undefined ? fields.priority : 1;
    this.focusLevel = fields.focus_level !== undefined ? fields.focus_level : 1.0;
    this.duration = fields.duration !== undefined ? fields.duration : 0.5;
    this.mode = fields.mode || 'HIGHLIGHT'; // HIGHLIGHT, DIM_BACKGROUND, SPOTLIGHT, ISOLATE, GLOW, MAGNIFY, POINTER, CUSTOM
    this.layer = fields.layer || 'foreground'; // background, normal, foreground, annotation, pointer
    this.visualEffect = fields.visual_effect || {}; // Opacity modifiers, blur, or desaturation values
    this.metadata = fields.metadata || {};
  }

  /**
   * Serializes the Focus instance to a pure JSON object.
   * @returns {object}
   */
  serialize() {
    return {
      focus_id: this.focusId,
      target: this.target,
      priority: this.priority,
      focus_level: this.focusLevel,
      duration: this.duration,
      mode: this.mode,
      layer: this.layer,
      visual_effect: this.visualEffect,
      metadata: this.metadata
    };
  }

  /**
   * Deserializes a Focus instance from a JSON object.
   * @param {object} json 
   * @returns {Focus}
   */
  static deserialize(json) {
    if (!json) return null;
    return new Focus(json);
  }
}

module.exports = Focus;
