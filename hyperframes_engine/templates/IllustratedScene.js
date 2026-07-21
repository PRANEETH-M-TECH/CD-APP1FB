const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * IllustratedScene.js
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
      sceneTl.fromTo('#ill-canvas-${sId}', { opacity: 0 }, { opacity: 1, duration: 0.5 });
      if (document.querySelector('#ill-title-${sId}')) {
        sceneTl.fromTo('#ill-title-${sId}', { opacity: 0, y: -12 }, { opacity: 1, y: 0, duration: 0.4 }, '<');
      }

      const action_${sId} = "${data.animation_action || 'none'}";
      // Match RenderTreeNode output ids: el_comp_<sceneNo>_<idx>
      const elementsSelector_${sId} = '[id^="el_comp_${sId}_"]';
      const elementsList_${sId} = document.querySelectorAll(elementsSelector_${sId});
      
      if (elementsList_${sId}.length > 0) {
        if (action_${sId} === 'rise') {
          sceneTl.fromTo(elementsSelector_${sId}, { y: 100, opacity: 0 }, { y: 0, opacity: 1, stagger: 0.1, duration: 0.8, ease: 'power2.out' });
        } else if (action_${sId} === 'fall') {
          sceneTl.fromTo(elementsSelector_${sId}, { y: -100, opacity: 0 }, { y: 0, opacity: 1, stagger: 0.1, duration: 0.8, ease: 'power2.out' });
        } else if (action_${sId} === 'spin') {
          sceneTl.fromTo(elementsSelector_${sId}, { rotation: 0, opacity: 0 }, { rotation: 360, opacity: 1, transformOrigin: '50% 50%', stagger: 0.1, duration: 1.0 });
        } else if (action_${sId} === 'scale_up') {
          sceneTl.fromTo(elementsSelector_${sId}, { scale: 0, opacity: 0 }, { scale: 1, opacity: 1, transformOrigin: '50% 50%', stagger: 0.1, duration: 0.6, ease: 'back.out(1.5)' });
        } else if (action_${sId} === 'slide_left') {
          sceneTl.fromTo(elementsSelector_${sId}, { x: 150, opacity: 0 }, { x: 0, opacity: 1, stagger: 0.1, duration: 0.6 });
        } else if (action_${sId} === 'slide_right') {
          sceneTl.fromTo(elementsSelector_${sId}, { x: -150, opacity: 0 }, { x: 0, opacity: 1, stagger: 0.1, duration: 0.6 });
        } else {
          sceneTl.fromTo(elementsSelector_${sId}, { opacity: 0 }, { opacity: 1, stagger: 0.1, duration: 0.5 });
        }
      }
    `;
  }
};
