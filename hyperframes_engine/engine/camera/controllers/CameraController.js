/**
 * CameraController.js
 * Controller responsible for translating Camera engine states into browser CSS transforms
 * relative to the viewport center.
 */
class CameraController {
  constructor(camera) {
    if (!camera) {
      throw new Error("[CameraController Error] Camera instance is required.");
    }
    this.camera = camera;
  }

  /**
   * Generates the CSS transform rule aligning the camera position to viewport center.
   * Viewport Center coordinate: (viewport.width / 2, viewport.height / 2) -> (640, 360)
   * @returns {string} CSS styles string containing transform rules
   */
  getTransformStyle() {
    const px = this.camera.position.x;
    const py = this.camera.position.y;
    const z = this.camera.zoom;
    const r = this.camera.rotation;

    // Resolve viewport offsets
    const cx = this.camera.viewport.width / 2;
    const cy = this.camera.viewport.height / 2;

    const dx = cx - px;
    const dy = cy - py;

    return `transform-origin: ${px}px ${py}px; transform: translate(${dx}px, ${dy}px) scale(${z}) rotate(${r}deg);`;
  }
}

module.exports = CameraController;
