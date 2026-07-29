const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * GeoMarker.js
 * Template renderer and animator for geography/historical location map markers.
 */
module.exports = {
  render: (sId, data, storyboard) => {
    const sceneJson = storyboard.scenes.find(s => s.scene_no === sId);
    const scene = Scene.deserialize(sceneJson);
    return Renderer.renderScene(scene);
  },

  animate: (sId, data) => {
    return `
      sceneTl.fromTo('#geo-title-${sId}', { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.4 });
      sceneTl.fromTo('#geo-markers-${sId} .geo-pin-wrapper', { opacity: 0, y: -30, scale: 0 }, { opacity: 1, y: 0, scale: 1, stagger: 0.25, duration: 0.6, ease: 'back.out(1.6)' }, 0.3);
    `;
  }
};
