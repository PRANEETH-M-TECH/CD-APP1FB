/**
 * Narration.js
 * Decoupled, serializable Narration model representing visual audio timelines,
 * cue timings, speakers, and segments.
 */
class Narration {
  constructor(fields = {}) {
    this.narrationId = fields.narration_id || `narr_${Math.random().toString(36).substr(2, 9)}`;
    this.speaker = fields.speaker || 'Narrator';
    this.language = fields.language || 'en';
    this.segments = fields.segments || []; // Array of NarrationSegment objects
    this.timing = fields.timing || {};
    this.pausePoints = fields.pause_points || [];
    this.cuePoints = fields.cue_points || [];
    this.metadata = fields.metadata || {};
  }

  /**
   * Serializes the Narration instance to a pure JSON object.
   * @returns {object}
   */
  serialize() {
    return {
      narration_id: this.narrationId,
      speaker: this.speaker,
      language: this.language,
      segments: this.segments.map(s => s.serialize()),
      timing: this.timing,
      pause_points: this.pausePoints,
      cue_points: this.cuePoints,
      metadata: this.metadata
    };
  }

  /**
   * Deserializes a Narration instance from JSON.
   * @param {object} json 
   * @returns {Narration}
   */
  static deserialize(json) {
    if (!json) return null;
    const NarrationSegment = require('./NarrationSegment');
    return new Narration({
      ...json,
      segments: (json.segments || []).map(s => NarrationSegment.deserialize(s))
    });
  }
}

module.exports = Narration;
