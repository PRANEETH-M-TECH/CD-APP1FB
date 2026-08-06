const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * MathDerivation.js
 * Template orchestrator delegating layout rendering entirely to the engine Renderer.
 */
module.exports = {
  render: (sId, data, storyboard) => {
    const sceneJson = storyboard.scenes.find(s => s.scene_no === sId);
    const scene = Scene.deserialize(sceneJson);
    return Renderer.renderScene(scene);
  },
  animate: (sId, data) => {
    const stepCount = (data.steps || []).length;
    return `
      sceneTl.fromTo('#math-title-${sId}', { opacity: 0, y: -15 }, { opacity: 1, y: 0, duration: 0.5 });
      ${data.formula ? `
        sceneTl.fromTo('#math-formula-${sId}', { opacity: 0, scale: 0.6 }, { opacity: 1, scale: 1, duration: 0.5, ease: 'back.out(1.4)' }, 0.2);
      ` : ''}

      // Reveal one step at a time (not a uniform stagger) and grow a real
      // connector between each consecutive pair once both are on screen -
      // this is what makes it read as a worked derivation flowing downward
      // rather than a stack of cards that happened to arrive staggered.
      const stepCount_${sId} = ${stepCount};
      const stepsContainer_${sId} = document.getElementById('math-steps-${sId}');
      let prevStepEl_${sId} = null;
      for (let i = 0; i < stepCount_${sId}; i++) {
        const stepEl = document.getElementById('math-step-${sId}-' + i);
        const badgeEl = document.getElementById('math-badge-${sId}-' + i);
        if (!stepEl) continue;
        const revealStart = 0.5 + i * 0.55;
        sceneTl.fromTo(stepEl, { opacity: 0, scale: 0.85, y: 15 }, { opacity: 1, scale: 1, y: 0, duration: 0.4, ease: 'power2.out' }, revealStart);
        if (badgeEl) {
          sceneTl.fromTo(badgeEl, { scale: 0 }, { scale: 1, duration: 0.35, ease: 'back.out(2)' }, revealStart + 0.1);
        }

        if (prevStepEl_${sId} && stepsContainer_${sId}) {
          const connector = document.createElement('div');
          connector.style.position = 'absolute';
          connector.style.width = '2px';
          connector.style.background = theme.accentColor;
          connector.style.opacity = '0';
          connector.style.left = '24px';
          connector.style.zIndex = '1';
          if (getComputedStyle(stepsContainer_${sId}).position === 'static') {
            stepsContainer_${sId}.style.position = 'relative';
          }
          const prevRect = prevStepEl_${sId}.getBoundingClientRect();
          const containerRect = stepsContainer_${sId}.getBoundingClientRect();
          const top = prevRect.bottom - containerRect.top;
          connector.style.top = top + 'px';
          connector.style.height = '0px';
          stepsContainer_${sId}.appendChild(connector);
          sceneTl.to(connector, { height: '12px', opacity: 0.8, duration: 0.3 }, revealStart - 0.15);
        }
        prevStepEl_${sId} = stepEl;
      }
    `;
  }
};
