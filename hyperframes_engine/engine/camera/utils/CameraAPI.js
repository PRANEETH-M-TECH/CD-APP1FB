const Camera = require('../models/Camera');

/**
 * CameraAPI.js
 * Developer-facing API to manipulate and modify Camera states programmatically
 * in a renderer-independent manner.
 */
module.exports = {
  /**
   * Instantiates a new Camera.
   * @param {object} fields 
   * @returns {Camera}
   */
  createCamera: (fields) => {
    return new Camera(fields);
  },

  /**
   * Updates camera look-at coordinates.
   * @param {Camera} camera 
   * @param {number} x 
   * @param {number} y 
   */
  setPosition: (camera, x, y) => {
    camera.position = { x, y };
  },

  /**
   * Updates camera zoom factor.
   * @param {Camera} camera 
   * @param {number} zoom 
   */
  setZoom: (camera, zoom) => {
    camera.zoom = zoom;
  },

  /**
   * Updates camera target focus.
   * @param {Camera} camera 
   * @param {string|object} target 
   */
  setTarget: (camera, target) => {
    camera.target = target;
  },

  /**
   * Updates camera viewport size bounds.
   * @param {Camera} camera 
   * @param {number} width 
   * @param {number} height 
   */
  setViewport: (camera, width, height) => {
    camera.viewport = { width, height };
  },

  /**
   * Serializes a camera object.
   * @param {Camera} camera 
   * @returns {object}
   */
  serialize: (camera) => {
    return camera.serialize();
  },

  /**
   * Deserializes a camera object.
   * @param {object} json 
   * @returns {Camera}
   */
  deserialize: (json) => {
    return Camera.deserialize(json);
  }
};
