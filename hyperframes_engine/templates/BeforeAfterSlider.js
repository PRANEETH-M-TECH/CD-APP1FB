const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * BeforeAfterSlider.js
 * Template renderer and animator for Cause vs Effect / Before vs After wipe transitions.
 */
module.exports = {
  render: (sId, data, storyboard) => {
    const sceneJson = storyboard.scenes.find(s => s.scene_no === sId);
    const scene = Scene.deserialize(sceneJson);
    return Renderer.renderScene(scene);
  },

  animate: (sId, data) => {
    return `
      sceneTl.fromTo('#ba-title-${sId}', { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.4 });
      sceneTl.fromTo('#ba-before-${sId}', { opacity: 0, x: -50 }, { opacity: 1, x: 0, duration: 0.5, ease: 'power2.out' }, 0.2);
      sceneTl.fromTo('#ba-divider-${sId}', { scale: 0, rotation: -180 }, { scale: 1, rotation: 0, duration: 0.4, ease: 'back.out(1.8)' }, 0.5);
      sceneTl.fromTo('#ba-after-${sId}', { opacity: 0, x: 50 }, { opacity: 1, x: 0, duration: 0.5, ease: 'power2.out' }, 0.6);
    `;
  }
};
