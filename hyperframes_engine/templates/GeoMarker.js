const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * GeoMarker.js
 * Template renderer and animator for geography/historical location map markers.
 */
module.exports = {
  render: (sId, data, storyboard) => {
    const title = data.title || 'Geographical Overview';
    const markers = data.markers || [{ label: 'Location A', x: 35, y: 45, description: 'Key historical site' }, { label: 'Location B', x: 65, y: 55, description: 'Major landmark' }];

    return `
      <div class="geomarker-container" id="geo-${sId}" style="width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px; position: relative;">
        <h2 class="theme-text" style="font-size: 34px; font-weight: 900; margin-bottom: 24px;" id="geo-title-${sId}">${title}</h2>

        <div class="map-viewport theme-card-bg theme-card-border" style="position: relative; width: 85%; height: 420px; border-radius: 24px; overflow: hidden; background: radial-gradient(circle, rgba(15,23,42,0.9) 0%, rgba(9,13,22,1) 100%);">
          <!-- Stylized Map Grid Overlay -->
          <svg viewBox="0 0 100 100" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; opacity: 0.15;">
            <pattern id="map-grid" width="10" height="10" patternUnits="userSpaceOnUse">
              <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#ffffff" stroke-width="0.5" />
            </pattern>
            <rect width="100" height="100" fill="url(#map-grid)" />
          </svg>

          <!-- Pins & Description Cards -->
          <div id="geo-markers-${sId}">
            ${markers.map((m, mIdx) => `
              <div class="geo-pin-wrapper" id="geo-pin-${sId}-${mIdx}" style="position: absolute; left: ${m.x || 50}%; top: ${m.y || 50}%; transform: translate(-50%, -50%); display: flex; flex-direction: column; align-items: center; z-index: 5;">
                <div class="pin-head theme-accent-bg" style="width: 18px; height: 18px; border-radius: 50%; box-shadow: 0 0 16px #38bdf8; position: relative;">
                  <div style="position: absolute; width: 100%; height: 100%; border-radius: 50%; border: 2px solid #38bdf8; animation: ping 2s infinite;"></div>
                </div>
                <div class="pin-card theme-card-bg theme-card-border" style="margin-top: 10px; padding: 10px 16px; border-radius: 12px; white-space: nowrap; text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.6);">
                  <div style="font-weight: 800; font-size: 15px; color: #ffffff;">${m.label || 'Marker'}</div>
                  ${m.description ? `<div style="font-size: 12px; color: rgba(255,255,255,0.7); margin-top: 2px;">${m.description}</div>` : ''}
                </div>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `;
  },

  animate: (sId, data) => {
    return `
      sceneTl.fromTo('#geo-title-${sId}', { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.4 });
      sceneTl.fromTo('#geo-markers-${sId} .geo-pin-wrapper', { opacity: 0, y: -30, scale: 0 }, { opacity: 1, y: 0, scale: 1, stagger: 0.25, duration: 0.6, ease: 'back.out(1.6)' }, 0.3);
    `;
  }
};
