/**
 * LayoutConstraints.js
 * Describes size limitations, preferred aspect ratios, anchor alignment mappings,
 * and container limits for Scene Graph components.
 */
class LayoutConstraints {
  constructor(fields = {}) {
    this.minWidth = fields.min_width || 0;
    this.maxWidth = fields.max_width || 99999;
    this.minHeight = fields.min_height || 0;
    this.maxHeight = fields.max_height || 99999;
    this.aspectRatio = fields.aspect_ratio || null; // e.g. 16/9, 4/3
    this.anchor = fields.anchor || null; // e.g. { x: 0.5, y: 0.5 }
  }

  /**
   * Serializes the constraints.
   * @returns {object}
   */
  serialize() {
    return {
      min_width: this.minWidth,
      max_width: this.maxWidth,
      min_height: this.minHeight,
      max_height: this.maxHeight,
      aspect_ratio: this.aspectRatio,
      anchor: this.anchor
    };
  }

  /**
   * Deserializes LayoutConstraints from JSON.
   * @param {object} json 
   * @returns {LayoutConstraints}
   */
  static deserialize(json) {
    if (!json) return null;
    return new LayoutConstraints(json);
  }
}

module.exports = LayoutConstraints;
