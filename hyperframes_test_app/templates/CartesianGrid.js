const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * CartesianGrid.js
 * Template renderer and animator for 2D coordinate grid, functions, and plot points.
 */
module.exports = {
  render: (sId, data, storyboard) => {
    const title = data.title || 'Coordinate Geometry';
    const eqLabel = data.equation_label || 'y = f(x)';
    const points = data.points || [{ x: 0, y: 0, label: 'Origin' }, { x: 3, y: 4, label: 'P(3,4)' }];

    return `
      <div class="cartesian-container" id="cartesian-${sId}" style="width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px; position: relative;">
        <h2 class="theme-text" style="font-size: 32px; font-weight: 900; margin-bottom: 12px;" id="cartesian-title-${sId}">${title}</h2>
        <div class="equation-badge theme-card-bg theme-card-border" id="cartesian-eq-${sId}" style="padding: 8px 20px; border-radius: 12px; font-weight: 700; font-size: 18px; margin-bottom: 20px; color: #38bdf8;">
          ${eqLabel}
        </div>

        <div class="grid-viewport theme-card-bg theme-card-border" style="position: relative; width: 680px; height: 380px; border-radius: 20px; overflow: hidden; display: flex; align-items: center; justify-content: center;">
          <!-- SVG Axis & Curves -->
          <svg viewBox="0 0 680 380" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0;">
            <!-- Grid Lines -->
            <line x1="0" y1="190" x2="680" y2="190" stroke="rgba(255,255,255,0.3)" stroke-width="2" />
            <line x1="340" y1="0" x2="340" y2="380" stroke="rgba(255,255,255,0.3)" stroke-width="2" />
            
            <!-- Curve Plot Line -->
            <path id="cartesian-curve-${sId}" d="M 100 300 Q 340 50 580 300" stroke="#38bdf8" stroke-width="4" fill="none" stroke-linecap="round" />
          </svg>

          <!-- Labeled Points -->
          <div id="cartesian-points-${sId}">
            ${points.map((p, pIdx) => `
              <div class="grid-point theme-accent-bg" id="cart-point-${sId}-${pIdx}" style="position: absolute; width: 14px; height: 14px; border-radius: 50%; left: ${340 + (p.x || 0) * 40}px; top: ${190 - (p.y || 0) * 30}px; transform: translate(-50%, -50%); box-shadow: 0 0 12px #38bdf8;">
                <span style="position: absolute; top: -24px; left: 50%; transform: translateX(-50%); font-size: 13px; font-weight: 800; white-space: nowrap; color: #ffffff; text-shadow: 0 2px 6px #000;">${p.label || ''}</span>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `;
  },

  animate: (sId, data) => {
    return `
      sceneTl.fromTo('#cartesian-title-${sId}', { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.4 });
      sceneTl.fromTo('#cartesian-eq-${sId}', { opacity: 0, scale: 0.8 }, { opacity: 1, scale: 1, duration: 0.4 }, 0.2);
      sceneTl.fromTo('#cartesian-curve-${sId}', { strokeDasharray: 800, strokeDashoffset: 800 }, { strokeDashoffset: 0, duration: 1.0, ease: 'power2.out' }, 0.4);
      sceneTl.fromTo('#cartesian-points-${sId} .grid-point', { scale: 0, opacity: 0 }, { scale: 1, opacity: 1, stagger: 0.2, duration: 0.4, ease: 'back.out(1.8)' }, 0.8);
    `;
  }
};
