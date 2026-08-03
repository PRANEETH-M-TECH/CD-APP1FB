const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * CycleTemplate.js
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
    const stages = data.stages || [];
    const total = stages.length;
    const timeStep = (dur - 1.5) / Math.max(1, total);
    return `
      sceneTl.fromTo('#cycle-title-${sId}', { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.6 }, 0.2);
      
      const circlePath_${sId} = document.getElementById('cycle-circle-path-${sId}');
      if (circlePath_${sId}) {
        const length = 2 * Math.PI * 140;
        gsap.set(circlePath_${sId}, { strokeDasharray: length, strokeDashoffset: length });
        sceneTl.to(circlePath_${sId}, { strokeDashoffset: 0, duration: ${dur.toFixed(1)}, ease: 'none' }, 0.4);
      }

      const totalStages_${sId} = ${total};
      const stagesData_${sId} = ${(JSON.stringify(stages))};
      const timeStep_${sId} = ${timeStep};

      stagesData_${sId}.forEach((stage, idx) => {
        const angle = (idx * (2 * Math.PI)) / totalStages_${sId} - Math.PI / 2;
        const x = 200 + 140 * Math.cos(angle);
        const y = 200 + 140 * Math.sin(angle);
        
        const stageEl = document.getElementById('cycle-stage-${sId}-' + idx);
        if (stageEl) {
          stageEl.style.left = x + 'px';
          stageEl.style.top = y + 'px';
          // See ConceptDiagram.js for why: centering + scale must both go through
          // GSAP (xPercent/yPercent + scale), not a hand-written 'transform' string,
          // or the element gets stuck at scale(0) - invisible - despite its opacity
          // tween completing.
          gsap.set(stageEl, { xPercent: -50, yPercent: -50, scale: 0 });

          const revealStart = 0.5 + (idx * timeStep_${sId});

          sceneTl.to(stageEl, {
            scale: 1,
            duration: 0.5,
            ease: 'back.out(1.6)'
          }, revealStart);

          // Pulse active stage
          sceneTl.to(stageEl, {
            boxShadow: '0 0 20px ' + theme.accentColor,
            duration: 0.4,
            yoyo: true,
            repeat: 1
          }, revealStart + 0.4);
        }
      });

      const orbitDot_${sId} = document.getElementById('cycle-orbit-dot-${sId}');
      if (orbitDot_${sId}) {
        const rotateObj = { angle: -90 };
        sceneTl.to(rotateObj, {
          angle: 270,
          repeat: 10,
          duration: 4,
          ease: 'none',
          onUpdate: () => {
            const rad = (rotateObj.angle * Math.PI) / 180;
            orbitDot_${sId}.setAttribute('cx', 200 + 140 * Math.cos(rad));
            orbitDot_${sId}.setAttribute('cy', 200 + 140 * Math.sin(rad));
          }
        }, 1.2);
      }
    `;
  }
};
