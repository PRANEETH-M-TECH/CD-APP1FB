const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * BeforeAfterSlider.js
 * Template renderer and animator for Cause vs Effect / Before vs After wipe transitions.
 */
module.exports = {
  render: (sId, data, storyboard) => {
    const title = data.title || 'Before vs After State';
    const before = data.before || { label: 'BEFORE', bullets: ['Initial state', 'Base conditions'] };
    const after = data.after || { label: 'AFTER', bullets: ['Transformed state', 'Resulting condition'] };

    return `
      <div class="before-after-container" id="ba-${sId}" style="width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px; position: relative;">
        <h2 class="theme-text" style="font-size: 34px; font-weight: 900; margin-bottom: 24px;" id="ba-title-${sId}">${title}</h2>

        <div class="ba-grid" style="display: flex; gap: 30px; width: 85%; height: 420px; position: relative; align-items: center; justify-content: center;">
          <!-- Before Card -->
          <div class="ba-card theme-card-bg theme-card-border" id="ba-before-${sId}" style="flex: 1; height: 100%; border-radius: 24px; padding: 30px; display: flex; flex-direction: column; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border-left: 4px solid #ef4444;">
            <div style="font-size: 22px; font-weight: 900; color: #ef4444; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 1px;">${before.label || 'BEFORE'}</div>
            <div style="display: flex; flex-direction: column; gap: 12px;">
              ${(before.bullets || []).map((b, bIdx) => `
                <div style="font-size: 16px; font-weight: 600; color: rgba(255,255,255,0.9); display: flex; align-items: center; gap: 10px;">
                  <span style="width: 8px; height: 8px; border-radius: 50%; background: #ef4444; flex-shrink: 0;"></span>
                  <span>${b}</span>
                </div>
              `).join('')}
            </div>
          </div>

          <!-- Divider Indicator -->
          <div id="ba-divider-${sId}" style="width: 50px; height: 50px; border-radius: 50%; background: #090d16; border: 2px solid rgba(255,255,255,0.2); display: flex; align-items: center; justify-content: center; font-weight: 900; z-index: 10; color: #38bdf8;">
            ➔
          </div>

          <!-- After Card -->
          <div class="ba-card theme-card-bg theme-card-border" id="ba-after-${sId}" style="flex: 1; height: 100%; border-radius: 24px; padding: 30px; display: flex; flex-direction: column; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border-left: 4px solid #22c55e;">
            <div style="font-size: 22px; font-weight: 900; color: #22c55e; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 1px;">${after.label || 'AFTER'}</div>
            <div style="display: flex; flex-direction: column; gap: 12px;">
              ${(after.bullets || []).map((b, bIdx) => `
                <div style="font-size: 16px; font-weight: 600; color: rgba(255,255,255,0.9); display: flex; align-items: center; gap: 10px;">
                  <span style="width: 8px; height: 8px; border-radius: 50%; background: #22c55e; flex-shrink: 0;"></span>
                  <span>${b}</span>
                </div>
              `).join('')}
            </div>
          </div>
        </div>
      </div>
    `;
  },

  animate: (sId, data) => {
    return `
      sceneTl.fromTo('#ba-title-${sId}', { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.4 });
      sceneTl.fromTo('#ba-before-${sId}', { opacity: 0, x: -50 }, { opacity: 1, x: 0, duration: 0.5, ease: 'power2.out' }, 0.2);
      sceneTl.fromTo('#ba-divider-${sId}', { scale: 0, rotation: -180 }, { scale: 1, rotation: 0, duration: 0.4, ease: 'back.out(1.8)' }, 0.5);
      sceneTl.fromTo('#ba-after-${sId}', { opacity: 0, x: 50 }, { opacity: 1, x: 0, duration: 0.5, ease: 'power2.out' }, 0.6);
    `;
  }
};
