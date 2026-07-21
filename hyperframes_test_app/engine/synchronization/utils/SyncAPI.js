const Narration = require('../models/Narration');
const SyncEvent = require('../models/SyncEvent');
const SyncManager = require('../manager/SyncManager');

/**
 * SyncAPI.js
 * Developer-facing public API to sync audio cues and narration segments.
 */
module.exports = {
  /**
   * Instantiates a new Narration cue timeline.
   * @param {object} fields 
   * @returns {Narration}
   */
  createNarration: (fields) => {
    return new Narration(fields);
  },

  /**
   * Instantiates a new SyncEvent.
   * @param {object} fields 
   * @returns {SyncEvent}
   */
  createEvent: (fields) => {
    return new SyncEvent(fields);
  },

  /**
   * Loads a new SyncManager.
   * @returns {SyncManager}
   */
  loadSyncManager: () => {
    return new SyncManager();
  },

  /**
   * Serializes a synchronization model.
   * @param {object} obj 
   * @returns {object}
   */
  serialize: (obj) => {
    return obj.serialize();
  },

  /**
   * Deserializes a narration configuration.
   * @param {object} json 
   * @returns {Narration}
   */
  deserialize: (json) => {
    return Narration.deserialize(json);
  }
};
