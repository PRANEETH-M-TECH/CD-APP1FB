const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * ColumnComparison.js
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
      sceneTl.fromTo('#comparison-title-${sId}', { opacity: 0, y: -15 }, { opacity: 1, y: 0, duration: 0.5 });
      sceneTl.fromTo('#comparison-col-left-${sId}', { opacity: 0, x: -50 }, { opacity: 1, x: 0, duration: 0.5 }, 0.2);
      sceneTl.fromTo('#comparison-col-right-${sId}', { opacity: 0, x: 50 }, { opacity: 1, x: 0, duration: 0.5 }, 0.3);
      sceneTl.fromTo('#comparison-${sId} .comparison-bullet', { opacity: 0, x: -10 }, { opacity: 1, x: 0, stagger: 0.1, duration: 0.3 }, 0.5);
    `;
  }
};
