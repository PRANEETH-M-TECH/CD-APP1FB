const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * TaxonomyTree.js
 * Template renderer and animator for hierarchical taxonomy / classification trees.
 */
module.exports = {
  render: (sId, data, storyboard) => {
    const sceneJson = storyboard.scenes.find(s => s.scene_no === sId);
    const scene = Scene.deserialize(sceneJson);
    return Renderer.renderScene(scene);
  },

  animate: (sId, data) => {
    return `
      sceneTl.fromTo('#taxonomy-title-${sId}', { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.4 });
      sceneTl.fromTo('#tax-root-${sId}', { scale: 0, opacity: 0 }, { scale: 1, opacity: 1, duration: 0.5, ease: 'back.out(1.5)' }, 0.2);
      sceneTl.fromTo('#tax-branches-${sId} .branch-card', { opacity: 0, y: 30, scale: 0.8 }, { opacity: 1, y: 0, scale: 1, stagger: 0.15, duration: 0.5, ease: 'power2.out' }, 0.5);
    `;
  }
};
