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
    const leftCount = ((data.left_col || {}).bullets || []).length;
    const rightCount = ((data.right_col || {}).bullets || []).length;
    return `
      sceneTl.fromTo('#comparison-title-${sId}', { opacity: 0, y: -15 }, { opacity: 1, y: 0, duration: 0.5 });
      sceneTl.fromTo('#comparison-col-left-${sId}', { opacity: 0, x: -50 }, { opacity: 1, x: 0, duration: 0.5, ease: 'power2.out' }, 0.2);
      sceneTl.fromTo('#comparison-col-right-${sId}', { opacity: 0, x: 50 }, { opacity: 1, x: 0, duration: 0.5, ease: 'power2.out' }, 0.3);

      // Bullets pop in one at a time per column, each with its own
      // back-eased icon arrival, instead of one uniform x-slide stagger -
      // the two columns visibly race each other into place.
      const leftCount_${sId} = ${leftCount};
      const rightCount_${sId} = ${rightCount};
      for (let i = 0; i < leftCount_${sId}; i++) {
        const el = document.getElementById('comparison-bullet-l-${sId}-' + i);
        if (!el) continue;
        gsap.set(el, { opacity: 0, scale: 0.6 });
        sceneTl.to(el, { opacity: 1, scale: 1, duration: 0.4, ease: 'back.out(1.8)' }, 0.7 + i * 0.18);
      }
      for (let i = 0; i < rightCount_${sId}; i++) {
        const el = document.getElementById('comparison-bullet-r-${sId}-' + i);
        if (!el) continue;
        gsap.set(el, { opacity: 0, scale: 0.6 });
        sceneTl.to(el, { opacity: 1, scale: 1, duration: 0.4, ease: 'back.out(1.8)' }, 0.8 + i * 0.18);
      }
    `;
  }
};
