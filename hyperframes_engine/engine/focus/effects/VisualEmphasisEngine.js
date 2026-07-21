/**
 * VisualEmphasisEngine.js
 * Decoupled engine assigning styling visual filters (dim, blur, glow shadows, scale scaling)
 * directly to the components styling parameters before the Renderer starts compiling HTML templates.
 */
class VisualEmphasisEngine {
  /**
   * Modifies component style parameters dynamically to emphasize key visual elements.
   * @param {Scene} scene 
   */
  static applyEmphasis(scene) {
    if (!scene || !Array.isArray(scene.focuses) || scene.focuses.length === 0) {
      return;
    }

    const activeFocus = scene.focuses[0];
    const focusedIds = scene.focuses.map(f => f.target);

    scene.traverse((node) => {
      if (!node.component) return;

      const isFocused = focusedIds.includes(node.id);

      switch (activeFocus.mode.toUpperCase()) {
        case 'DIM_BACKGROUND':
          if (!isFocused) {
            node.component.style.opacity = 0.35;
            node.component.style.filter = 'blur(1.5px) grayscale(40%)';
          } else {
            node.component.style.opacity = 1.0;
          }
          break;

        case 'HIGHLIGHT':
          if (isFocused) {
            node.component.style.filter = 'drop-shadow(0 0 12px #3b82f6)';
            node.component.style.transform = 'scale(1.04)';
          }
          break;

        case 'GLOW':
          if (isFocused) {
            node.component.style.filter = 'drop-shadow(0 0 20px #eab308)';
          }
          break;

        case 'ISOLATE':
          if (!isFocused) {
            node.component.style.visibility = 'hidden';
            node.component.visibility = false;
          }
          break;
      }
    });
  }
}

module.exports = VisualEmphasisEngine;
