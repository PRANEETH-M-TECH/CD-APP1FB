/**
 * SafeArea.js
 * Model representing safe viewport margin areas (title-safe and subtitle-safe margins).
 */
class SafeArea {
  constructor(fields = {}) {
    this.top = fields.top !== undefined ? fields.top : 40;
    this.left = fields.left !== undefined ? fields.left : 60;
    this.bottom = fields.bottom !== undefined ? fields.bottom : 40;
    this.right = fields.right !== undefined ? fields.right : 60;
  }

  /**
   * Serializes the SafeArea instance.
   * @returns {object}
   */
  serialize() {
    return {
      top: this.top,
      left: this.left,
      bottom: this.bottom,
      right: this.right
    };
  }

  /**
   * Deserializes a SafeArea instance.
   * @param {object} json 
   * @returns {SafeArea}
   */
  static deserialize(json) {
    if (!json) return null;
    return new SafeArea(json);
  }
}

module.exports = SafeArea;
