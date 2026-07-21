/**
 * PointerCalloutSystem.js
 * Decoupled pointer/marker architecture to inject leader bubbles or pointers
 * relative to scene targets.
 */
class PointerCalloutSystem {
  /**
   * Instantiates a pointer node referencing a Scene Graph target.
   * @param {string} sId 
   * @param {string} type 
   * @param {string} targetId 
   * @param {string} text 
   * @returns {SceneNode}
   */
  static createPointerNode(sId, type, targetId, text = '') {
    const SceneNode = require('../../scene/SceneNode');
    const ComponentFactory = require('../../components/factory/ComponentFactory');

    // Create annotation layout structure
    const component = ComponentFactory.createComponent({
      id: `pointer_${sId}`,
      type: 'CUSTOM',
      properties: {
        pointerType: type, // arrow, bubbles, labels, leader_lines
        target: targetId,
        text: text
      },
      style: {
        zIndex: 10000 // Pointer overlay layer
      }
    });

    return new SceneNode(`pointer_${sId}`, 'POINTER', component);
  }
}

module.exports = PointerCalloutSystem;
