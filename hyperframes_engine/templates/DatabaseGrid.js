const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * DatabaseGrid.js
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
      sceneTl.fromTo('#db-title-${sId}', { opacity: 0, y: -15 }, { opacity: 1, y: 0, duration: 0.5 });
      sceneTl.fromTo('#db-card-${sId}', { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out' }, 0.2);
      sceneTl.fromTo('tr[id^="db-row-${sId}"]', { opacity: 0, y: 15 }, { opacity: 1, y: 0, stagger: 0.1, duration: 0.3 }, 0.4);
    `;
  }
};
