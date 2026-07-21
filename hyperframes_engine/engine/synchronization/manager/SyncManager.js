const SyncEvent = require('../models/SyncEvent');

/**
 * SyncManager.js
 * Align synchronization events and tracks synchronization indices.
 */
class SyncManager {
  constructor() {
    this.events = [];
  }

  /**
   * Registers a timeline SyncEvent.
   * @param {SyncEvent} evt 
   */
  registerEvent(evt) {
    if (evt) {
      this.events.push(evt);
      this.alignTimelines();
    }
  }

  /**
   * Sorts synchronization events sequentially by timestamp.
   */
  alignTimelines() {
    this.events.sort((a, b) => a.timestamp - b.timestamp);
  }

  /**
   * Resolves events occurring at a specific time marker.
   * @param {number} timestamp 
   * @returns {Array<SyncEvent>}
   */
  getEventsAt(timestamp) {
    return this.events.filter(e => Math.abs(e.timestamp - timestamp) < 0.05);
  }
}

module.exports = SyncManager;
