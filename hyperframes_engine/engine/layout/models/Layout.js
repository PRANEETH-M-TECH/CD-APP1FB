/**
 * Layout.js
 * Extensible, serializable Layout model representing sizing, margins, paddings, alignment
 * guidelines, and constraints of a scene's components.
 */
class Layout {
  constructor(fields = {}) {
    this.layoutId = fields.layout_id || `lay_${Math.random().toString(36).substr(2, 9)}`;
    this.layoutType = fields.layout_type || 'FREE'; // FREE, GRID, ROW, COLUMN, STACK, FLOW, ABSOLUTE, ANCHOR, CUSTOM
    this.constraints = fields.constraints || null;
    this.alignment = fields.alignment || 'CENTER'; // LEFT, CENTER, RIGHT, TOP, BOTTOM, MIDDLE, STRETCH
    this.padding = fields.padding || { top: 0, left: 0, bottom: 0, right: 0 };
    this.margin = fields.margin || { top: 0, left: 0, bottom: 0, right: 0 };
    this.spacing = fields.spacing !== undefined ? fields.spacing : 0;
    this.safeArea = fields.safe_area || null;
    this.responsive = fields.responsive !== undefined ? fields.responsive : true;
    this.metadata = fields.metadata || {};
  }

  /**
   * Serializes the Layout instance to a pure JSON object.
   * @returns {object}
   */
  serialize() {
    return {
      layout_id: this.layoutId,
      layout_type: this.layoutType,
      constraints: this.constraints ? this.constraints.serialize() : null,
      alignment: this.alignment,
      padding: this.padding,
      margin: this.margin,
      spacing: this.spacing,
      safe_area: this.safeArea ? this.safeArea.serialize() : null,
      responsive: this.responsive,
      metadata: this.metadata
    };
  }

  /**
   * Deserializes a Layout instance from a JSON object.
   * @param {object} json 
   * @returns {Layout}
   */
  static deserialize(json) {
    if (!json) return null;
    const LayoutConstraints = require('../constraints/LayoutConstraints');
    const SafeArea = require('./SafeArea');
    return new Layout({
      ...json,
      constraints: json.constraints ? LayoutConstraints.deserialize(json.constraints) : null,
      safe_area: json.safe_area ? SafeArea.deserialize(json.safe_area) : null
    });
  }
}

module.exports = Layout;
