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
    const markerCount = (data.markers || []).length;
    return `
      sceneTl.fromTo('#geo-title-${sId}', { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.4 });
      sceneTl.fromTo('#geo-markers-${sId} .geo-pin-wrapper', { opacity: 0, y: -30, scale: 0 }, { opacity: 1, y: 0, scale: 1, stagger: 0.25, duration: 0.6, ease: 'back.out(1.6)' }, 0.3);

      // Each pin gets a continuous "map ping" ring pulse once it lands,
      // staggered so they don't all pulse in unison - a still map with pins
      // dropped on it doesn't read as alive the way a subtly pinging one does.
      const markerCount_${sId} = ${markerCount};
      for (let i = 0; i < markerCount_${sId}; i++) {
        const pin = document.querySelectorAll('#geo-markers-${sId} .geo-pin-wrapper')[i];
        if (!pin) continue;
        const ring = document.createElement('div');
        ring.style.position = 'absolute';
        ring.style.top = '0';
        ring.style.left = '50%';
        ring.style.width = '18px';
        ring.style.height = '18px';
        ring.style.marginLeft = '-9px';
        ring.style.borderRadius = '50%';
        ring.style.border = '2px solid ' + theme.accentColor;
        ring.style.pointerEvents = 'none';
        if (getComputedStyle(pin).position === 'static') pin.style.position = 'relative';
        pin.appendChild(ring);
        gsap.set(ring, { scale: 1, opacity: 0.9 });
        sceneTl.to(ring, { scale: 2.4, opacity: 0, duration: 1.4, repeat: 240, ease: 'power1.out' }, 0.9 + i * 0.25);
      }
    `;
  }
};
