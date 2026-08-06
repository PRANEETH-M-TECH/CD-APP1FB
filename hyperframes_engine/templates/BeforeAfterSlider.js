const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * BeforeAfterSlider.js
 * Template renderer and animator for Cause vs Effect / Before vs After wipe transitions.
 */
module.exports = {
  render: (sId, data, storyboard) => {
    const sceneJson = storyboard.scenes.find(s => s.scene_no === sId);
    const scene = Scene.deserialize(sceneJson);
    return Renderer.renderScene(scene);
  },

  animate: (sId, data) => {
    return `
      sceneTl.fromTo('#ba-title-${sId}', { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.4 });
      sceneTl.fromTo('#ba-before-${sId}', { opacity: 0, x: -50 }, { opacity: 1, x: 0, duration: 0.5, ease: 'power2.out' }, 0.2);
      sceneTl.fromTo('#ba-divider-${sId}', { scale: 0, rotation: -180 }, { scale: 1, rotation: 0, duration: 0.4, ease: 'back.out(1.8)' }, 0.5);

      // The "after" side wipes in via a clip-path reveal from the divider
      // outward, like a real before/after transition, instead of just
      // sliding in the same way the "before" side did - this is the one
      // moment in the scene that should visually read as a change happening.
      const afterEl_${sId} = document.getElementById('ba-after-${sId}');
      if (afterEl_${sId}) {
        gsap.set(afterEl_${sId}, { opacity: 1, x: 0, clipPath: 'inset(0 100% 0 0)' });
        sceneTl.to(afterEl_${sId}, { clipPath: 'inset(0 0% 0 0)', duration: 0.7, ease: 'power2.inOut' }, 0.75);
      }

      // Divider keeps a slow pulse for the rest of the scene, marking the
      // "boundary" between the two states even after the transition lands.
      sceneTl.to('#ba-divider-${sId}', {
        boxShadow: '0 0 18px ' + theme.accentColor,
        scale: 1.12,
        duration: 0.8,
        yoyo: true,
        repeat: 240,
        ease: 'sine.inOut'
      }, 1.5);
    `;
  }
};
