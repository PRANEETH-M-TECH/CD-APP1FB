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
      sceneTl.fromTo('#venn-content-right-${sId} .venn-item-card', { opacity: 0, scale: 0.5 }, { opacity: 1, scale: 1, stagger: 0.1, duration: 0.4 }, 0.7);

      // The overlap is the whole point of a Venn scene, so it gets a distinct
      // arrival - a beat later than either side, with a brighter pop and its
      // own glow - instead of appearing on the same uniform stagger as
      // everything else.
      sceneTl.fromTo('#venn-content-middle-${sId} .venn-item-card', { opacity: 0, scale: 0.3 }, { opacity: 1, scale: 1.08, stagger: 0.12, duration: 0.5, ease: 'back.out(2)' }, 1.0);
      sceneTl.to('#venn-content-middle-${sId} .venn-item-card', { scale: 1, boxShadow: '0 0 18px ' + theme.accentColor, duration: 0.3, yoyo: true, repeat: 3 }, 1.5);

      // Both circles breathe slowly for the rest of the scene so the diagram
      // doesn't go static the moment it's fully on screen.
      sceneTl.to('#venn-circle-left-${sId}', { scale: 1.04, duration: 1.6, yoyo: true, repeat: 240, ease: 'sine.inOut' }, 1.0);
      sceneTl.to('#venn-circle-right-${sId}', { scale: 1.04, duration: 1.6, yoyo: true, repeat: 240, ease: 'sine.inOut' }, 1.3);
    `;
  }
};
