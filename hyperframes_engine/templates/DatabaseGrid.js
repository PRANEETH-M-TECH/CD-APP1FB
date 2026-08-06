const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * DatabaseGrid.js
 * Template orchestrator delegating layout rendering entirely to the engine Renderer.
 */
module.exports = {
  render: (sId, data, storyboard) => {
    const sceneJson = storyboard.scenes.find(s => s.scene_no === sId);
    const scene = Scene.deserialize(sceneJson);
    return Renderer.renderScene(scene);
  },
  animate: (sId, data, storyboard, sceneDuration) => {
    const dur = sceneDuration || 8.0;
    const rowCount = (data.rows || []).length;
    const sweepStep = rowCount > 0 ? Math.max(0.3, Math.min(0.7, (dur - 2.0) / rowCount)) : 0.4;
    return `
      sceneTl.fromTo('#db-title-${sId}', { opacity: 0, y: -15 }, { opacity: 1, y: 0, duration: 0.5 });
      sceneTl.fromTo('#db-card-${sId}', { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out' }, 0.2);
      sceneTl.fromTo('tr[id^="db-row-${sId}"]', { opacity: 0, y: 15 }, { opacity: 1, y: 0, stagger: 0.1, duration: 0.3 }, 0.4);

      // A highlight sweeps row by row after the table has fully entered, as
      // if the narrator is pointing at the current row being discussed,
      // instead of the table just sitting static once it's on screen.
      const rowCount_${sId} = ${rowCount};
      const sweepStep_${sId} = ${sweepStep};
      const sweepStart_${sId} = 1.0;
      for (let r = 0; r < rowCount_${sId}; r++) {
        const rowEl = document.getElementById('db-row-${sId}-' + r);
        if (!rowEl) continue;
        const t = sweepStart_${sId} + r * sweepStep_${sId};
        sceneTl.to(rowEl, { backgroundColor: 'rgba(' + theme.accentRgb + ', 0.16)', duration: sweepStep_${sId} * 0.35 }, t);
        sceneTl.to(rowEl, { backgroundColor: 'rgba(0,0,0,0)', duration: sweepStep_${sId} * 0.35 }, t + sweepStep_${sId} * 0.5);
      }
    `;
  }
};
