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

  animate: (sId, data, storyboard, sceneDuration) => {
    const dur = sceneDuration || 8.0;
    const root = data.root || {};
    const total = (root.children || []).length;
    const timeStep = total > 0 ? Math.min(0.35, (dur - 2.5) / total) : 0.15;
    return `
      sceneTl.fromTo('#taxonomy-title-${sId}', { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.4 });
      sceneTl.fromTo('#tax-root-${sId}', { scale: 0, opacity: 0 }, { scale: 1, opacity: 1, duration: 0.5, ease: 'back.out(1.5)' }, 0.2);

      // Root gets a slow ambient pulse for the rest of the scene, not just a
      // one-shot entrance, so the origin of the tree stays visually alive
      // while each branch reveals below it.
      sceneTl.to('#tax-root-${sId}', {
        boxShadow: '0 14px 32px rgba(0,0,0,0.5), 0 0 22px ' + theme.accentColor,
        duration: 1.1,
        yoyo: true,
        repeat: 240
      }, 0.8);

      const branchTotal_${sId} = ${total};
      const timeStep_${sId} = ${timeStep};
      for (let idx = 0; idx < branchTotal_${sId}; idx++) {
        const branchEl = document.getElementById('tax-branch-${sId}-' + idx);
        if (!branchEl) continue;
        const revealStart = 0.6 + idx * timeStep_${sId};
        sceneTl.fromTo(branchEl, { opacity: 0, y: 30, scale: 0.8 }, { opacity: 1, y: 0, scale: 1, duration: 0.5, ease: 'power2.out' }, revealStart);
        // Brief highlight pulse on arrival, echoing the parent's glow, so each
        // branch reads as "handed down from the root" rather than appearing
        // independently.
        sceneTl.to(branchEl, {
          boxShadow: '0 8px 20px rgba(0,0,0,0.4), 0 0 16px ' + theme.accentColor,
          duration: 0.35,
          yoyo: true,
          repeat: 1
        }, revealStart + 0.3);
      }
    `;
  }
};
