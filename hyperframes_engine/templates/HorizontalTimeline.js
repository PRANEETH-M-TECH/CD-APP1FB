const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * HorizontalTimeline.js
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
      sceneTl.fromTo('#timeline-title-${sId}', { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.5 });
      sceneTl.fromTo('#timeline-active-line-${sId}', { attr: { x1: '5%', x2: '5%' } }, { attr: { x2: '95%' }, duration: 1.0, ease: 'power2.inOut' }, 0.3);
      
      const stagesData_${sId} = ${(JSON.stringify(data.stages || []))};
      stagesData_${sId}.forEach((stage, idx) => {
        const stageEl = document.getElementById('timeline-stage-${sId}-' + idx);
        if (stageEl) {
          sceneTl.fromTo(stageEl, { opacity: 0, scale: 0 }, { opacity: 1, scale: 1, duration: 0.5, ease: 'back.out(1.5)' }, 0.5 + idx * 0.15);
        }
      });
    `;
  }
};
