/**
 * Camera.js
 * Extensible, serializable Camera model representing viewing parameters of a scene.
 * Completely renderer-independent and layout-agnostic.
 */
class Camera {
  constructor(fields = {}) {
    this.cameraId = fields.camera_id || `cam_${Math.random().toString(36).substr(2, 9)}`;
    this.position = fields.position || { x: 640, y: 360 };
    this.target = fields.target || null; // Node/component ID or coordinate targets
    this.zoom = fields.zoom !== undefined ? fields.zoom : 1.0;
    this.rotation = fields.rotation !== undefined ? fields.rotation : 0.0;
    this.viewport = fields.viewport || { width: 1280, height: 720 };
    this.anchor = fields.anchor || { x: 0.5, y: 0.5 };
    this.padding = fields.padding || 0;
    this.mode = fields.mode || 'STATIC'; // STATIC, PAN, ZOOM, FOLLOW, FRAME_OBJECT, FIT_SCENE
    this.timeline = fields.timeline || []; // Timeline keyframe states
    this.metadata = fields.metadata || {};
    this.futureConstraints = fields.future_constraints || {};
  }

  /**
   * Serializes the Camera instance to a pure JSON object.
   * @returns {object}
   */
  serialize() {
    return {
      camera_id: this.cameraId,
      position: this.position,
      target: this.target,
      zoom: this.zoom,
      rotation: this.rotation,
      viewport: this.viewport,
      anchor: this.anchor,
      padding: this.padding,
      mode: this.mode,
      timeline: this.timeline,
      metadata: this.metadata,
      future_constraints: this.futureConstraints
    };
  }

  /**
   * Deserializes a Camera instance from a JSON object.
   * @param {object} json 
   * @returns {Camera}
   */
  static deserialize(json) {
    if (!json) return null;
    return new Camera(json);
  }
}

module.exports = Camera;
