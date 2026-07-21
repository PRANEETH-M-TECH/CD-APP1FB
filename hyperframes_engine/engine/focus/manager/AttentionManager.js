/**
 * AttentionManager.js
 * Decoupled manager sorting active focus priorities, resolving target conflicts,
 * and requesting camera track adjustments if necessary.
 */
class AttentionManager {
  /**
   * Sorts and resolves focus priorities for a scene graph.
   * If a high-priority focus target requests camera framing, updates scene camera boundaries.
   * @param {Scene} scene 
   */
  static resolveSceneFocus(scene) {
    if (!scene || !Array.isArray(scene.focuses) || scene.focuses.length === 0) {
      return;
    }

    // Sort focus configurations by priority descending
    const sortedFocuses = [...scene.focuses].sort((a, b) => b.priority - a.priority);
    const activeFocus = sortedFocuses[0];

    // If active focus requests camera framing tracking, update scene camera targets
    if (activeFocus && activeFocus.metadata && activeFocus.metadata.frame_camera && scene.camera) {
      const targetNode = scene.findNode(activeFocus.target);
      if (targetNode && targetNode.component) {
        // Resolve target absolute/styling coordinate position
        const x = parseFloat(targetNode.component.style.left || targetNode.component.properties.x || 640);
        const y = parseFloat(targetNode.component.style.top || targetNode.component.properties.y || 360);

        // Update Camera properties dynamically (Camera consumes final layout coordinates)
        scene.camera.position = { x, y };
        scene.camera.target = activeFocus.target;

        // Apply focus level as camera zoom bounds
        if (activeFocus.focusLevel) {
          scene.camera.zoom = activeFocus.focusLevel;
        }
      }
    }
  }
}

module.exports = AttentionManager;
