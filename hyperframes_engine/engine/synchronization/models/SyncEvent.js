/**
 * SyncEvent.js
 * Represents a timing synchronization event within the visual rendering timeline.
 */
class SyncEvent {
  constructor(fields = {}) {
    this.eventId = fields.event_id || `evt_${Math.random().toString(36).substr(2, 9)}`;
    this.type = fields.type || 'START_SEGMENT'; // START_SEGMENT, END_SEGMENT, START_CAMERA, SHOW_SUBTITLE, HIDE_SUBTITLE
    this.timestamp = fields.timestamp || 0;
    this.targetId = fields.target_id || null;
    this.value = fields.value || null;
    this.metadata = fields.metadata || {};
  }

  /**
   * Serializes the SyncEvent instance.
   * @returns {object}
   */
  serialize() {
    return {
      event_id: this.eventId,
      type: this.type,
      timestamp: this.timestamp,
      target_id: this.targetId,
      value: this.value,
      metadata: this.metadata
    };
  }

  /**
   * Deserializes a SyncEvent instance.
   * @param {object} json 
   * @returns {SyncEvent}
   */
  static deserialize(json) {
    if (!json) return null;
    return new SyncEvent(json);
  }
}

module.exports = SyncEvent;
