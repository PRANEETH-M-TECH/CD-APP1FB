const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * GeneralScene.js
 * Template orchestrator delegating layout rendering entirely to the engine Renderer.
 */
module.exports = {
  render: (sId, data, storyboard) => {
    const sceneJson = storyboard.scenes.find(s => s.scene_no === sId);
    const scene = Scene.deserialize(sceneJson);
    return Renderer.renderScene(scene);
  },
  animate: (sId, data) => {
    return `
      sceneTl.fromTo('#general-title-${sId}', { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.5 });
    `;
  }
};
