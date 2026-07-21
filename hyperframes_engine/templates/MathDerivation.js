const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * MathDerivation.js
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
      sceneTl.fromTo('#math-title-${sId}', { opacity: 0, y: -15 }, { opacity: 1, y: 0, duration: 0.5 });
      ${data.formula ? `
        sceneTl.fromTo('#math-formula-${sId}', { opacity: 0, scale: 0.6 }, { opacity: 1, scale: 1, duration: 0.5, ease: 'back.out(1.4)' }, 0.2);
      ` : ''}
      sceneTl.fromTo('#math-steps-${sId} .math-step-card', { opacity: 0, scale: 0.8, y: 15 }, { opacity: 1, scale: 1, y: 0, stagger: 0.2, duration: 0.4 }, 0.4);
    `;
  }
};
