/**
 * Subtitle.js
 * Decoupled subtitle segment capturing multi-line text lines and styling keys.
 */
class Subtitle {
  constructor(fields = {}) {
    this.subtitleId = fields.subtitle_id || `sub_${Math.random().toString(36).substr(2, 9)}`;
    this.text = fields.text || '';
    this.startTime = fields.start_time || 0;
    this.endTime = fields.end_time || 0;
    this.speaker = fields.speaker || '';
    this.language = fields.language || 'en';
    this.style = fields.style || {}; // Visual style configuration variables
  }

  /**
   * Serializes the Subtitle instance to a JSON object.
   * @returns {object}
   */
  serialize() {
    return {
      subtitle_id: this.subtitleId,
      text: this.text,
      start_time: this.startTime,
      end_time: this.endTime,
      speaker: this.speaker,
      language: this.language,
      style: this.style
    };
  }

  /**
   * Deserializes a Subtitle instance from JSON.
   * @param {object} json 
   * @returns {Subtitle}
   */
  static deserialize(json) {
    if (!json) return null;
    return new Subtitle(json);
  }
}

module.exports = Subtitle;
