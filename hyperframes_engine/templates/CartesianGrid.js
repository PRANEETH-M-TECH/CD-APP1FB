const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * CartesianGrid.js
 * Template renderer and animator for 2D coordinate grid, functions, and plot points.
 */
module.exports = {
  render: (sId, data, storyboard) => {
    const sceneJson = storyboard.scenes.find(s => s.scene_no === sId);
    const scene = Scene.deserialize(sceneJson);
    return Renderer.renderScene(scene);
  },

  animate: (sId, data) => {
    return `
      sceneTl.fromTo('#cartesian-title-${sId}', { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.4 });
      sceneTl.fromTo('#cartesian-eq-${sId}', { opacity: 0, scale: 0.8 }, { opacity: 1, scale: 1, duration: 0.4 }, 0.2);
      sceneTl.fromTo('#cartesian-curve-${sId}', { strokeDasharray: 800, strokeDashoffset: 800 }, { strokeDashoffset: 0, duration: 1.0, ease: 'power2.out' }, 0.4);
      sceneTl.fromTo('#cartesian-points-${sId} .grid-point', { scale: 0, opacity: 0 }, { scale: 1, opacity: 1, stagger: 0.2, duration: 0.4, ease: 'back.out(1.8)' }, 0.8);

      // Points keep a gentle continuous pulse once plotted, so the plot
      // doesn't go inert the instant the curve finishes drawing.
      sceneTl.to('#cartesian-points-${sId} .grid-point', {
        boxShadow: '0 0 16px #38bdf8',
        scale: 1.15,
        duration: 0.9,
        yoyo: true,
        repeat: 240,
        stagger: 0.15,
        ease: 'sine.inOut'
      }, 1.6);
    `;
  }
};
