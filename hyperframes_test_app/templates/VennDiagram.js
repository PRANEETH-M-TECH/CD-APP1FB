const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * VennDiagram.js
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
      sceneTl.fromTo('#venn-circle-left-${sId}', { opacity: 0, x: -100 }, { opacity: 0.45, x: -70, duration: 0.6, ease: 'power2.out' }, 0.2);
      sceneTl.fromTo('#venn-circle-right-${sId}', { opacity: 0, x: 100 }, { opacity: 0.45, x: 70, duration: 0.6, ease: 'power2.out' }, 0.2);
      sceneTl.fromTo('#venn-content-left-${sId} .venn-item-card', { opacity: 0, scale: 0.5 }, { opacity: 1, scale: 1, stagger: 0.1, duration: 0.4 }, 0.5);
      sceneTl.fromTo('#venn-content-middle-${sId} .venn-item-card', { opacity: 0, scale: 0.5 }, { opacity: 1, scale: 1, stagger: 0.1, duration: 0.4 }, 0.7);
      sceneTl.fromTo('#venn-content-right-${sId} .venn-item-card', { opacity: 0, scale: 0.5 }, { opacity: 1, scale: 1, stagger: 0.1, duration: 0.4 }, 0.9);
    `;
  }
};
