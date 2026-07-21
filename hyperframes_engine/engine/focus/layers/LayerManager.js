const LAYERS = {
  background: 1,
  normal: 10,
  foreground: 100,
  annotation: 1000,
  pointer: 10000
};

/**
 * LayerManager.js
 * Decoupled manager assigning browser z-indices based on abstract layered depth levels.
 */
class LayerManager {
  /**
   * Translates abstract layer names to numeric z-order depth bounds.
   * @param {string} layerName 
   * @returns {number}
   */
  static getZIndex(layerName) {
    if (!layerName) return LAYERS.normal;
    return LAYERS[layerName.toLowerCase()] || LAYERS.normal;
  }

  /**
   * Computes and assigns z-order boundaries on scene component style configurations.
   * @param {Scene} scene 
   */
  static applyLayering(scene) {
    if (!scene) return;

    const focuses = scene.focuses || [];

    scene.traverse((node) => {
      if (!node.component) return;

      // Default z-order layer
      let layer = 'normal';

      // Re-route to custom depth if target is linked to active focus
      const matchingFocus = focuses.find(f => f.target === node.id);
      if (matchingFocus) {
        layer = matchingFocus.layer;
      }

      // Update component styling properties directly
      node.component.style.zIndex = LayerManager.getZIndex(layer);
    });
  }
}

module.exports = LayerManager;
