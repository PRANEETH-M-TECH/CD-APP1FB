/**
 * NarrationSegment.js
 * Represents a single text segment within a narration block.
 * References logical concepts and Scene Graph entities instead of rendered nodes.
 */
class NarrationSegment {
  constructor(fields = {}) {
    this.segmentId = fields.segment_id || `seg_${Math.random().toString(36).substr(2, 9)}`;
    this.text = fields.text || '';
    this.startTime = fields.start_time || 0;
    this.estimatedDuration = fields.estimated_duration || 0;
    this.cueId = fields.cue_id || null;
    this.relatedTeachingStep = fields.related_teaching_step || null;
    this.relatedScene = fields.related_scene || null;
    this.subtitleReference = fields.subtitle_reference || null;
    this.metadata = fields.metadata || {};
  }

  /**
   * Serializes the NarrationSegment instance.
   * @returns {object}
   */
  serialize() {
    return {
      segment_id: this.segmentId,
      text: this.text,
      start_time: this.startTime,
      estimated_duration: this.estimatedDuration,
      cue_id: this.cueId,
      related_teaching_step: this.relatedTeachingStep,
      related_scene: this.relatedScene,
      subtitle_reference: this.subtitleReference,
      metadata: this.metadata
    };
  }

  /**
   * Deserializes a NarrationSegment instance.
   * @param {object} json 
   * @returns {NarrationSegment}
   */
  static deserialize(json) {
    if (!json) return null;
    return new NarrationSegment(json);
  }
}

module.exports = NarrationSegment;
