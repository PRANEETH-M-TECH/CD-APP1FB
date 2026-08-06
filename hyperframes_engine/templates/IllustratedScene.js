const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * IllustratedScene.js
 * Template orchestrator delegating layout rendering entirely to the engine Renderer.
 */
module.exports = {
  render: (sId, data, storyboard) => {
    const sceneJson = storyboard.scenes.find(s => s.scene_no === sId);
    const scene = Scene.deserialize(sceneJson);
    return Renderer.renderScene(scene);
  },
  animate: (sId, data) => {
    const elements = data.elements || [];
    return `
      // The canvas itself is always visible immediately - visibility is carried
      // entirely by the individual elements' own entrance animation below, not
      // by a second, separate wrapper-level fade this scene would otherwise
      // depend on completing first. One less thing that has to succeed in
      // sequence for content to actually show up.
      gsap.set('#ill-canvas-${sId}', { opacity: 1 });
      if (document.querySelector('#ill-title-${sId}')) {
        sceneTl.fromTo('#ill-title-${sId}', { opacity: 0, y: -12 }, { opacity: 1, y: 0, duration: 0.4 });
      }

      const action_${sId} = "${data.animation_action || 'none'}";
      const elementsMeta_${sId} = ${JSON.stringify(elements)};
      // Match RenderTreeNode output ids: el_comp_<sceneNo>_<idx>
      const elementsSelector_${sId} = '[id^="el_comp_${sId}_"]';
      const elementsList_${sId} = document.querySelectorAll(elementsSelector_${sId});

      if (elementsList_${sId}.length > 0) {
        // One-time entrance, per the LLM's chosen animation_action, same as before.
        if (action_${sId} === 'rise') {
          sceneTl.fromTo(elementsSelector_${sId}, { y: 100, opacity: 0 }, { y: 0, opacity: 1, stagger: 0.1, duration: 0.8, ease: 'power2.out' });
        } else if (action_${sId} === 'fall') {
          sceneTl.fromTo(elementsSelector_${sId}, { y: -100, opacity: 0 }, { y: 0, opacity: 1, stagger: 0.1, duration: 0.8, ease: 'power2.out' });
        } else if (action_${sId} === 'spin') {
          sceneTl.fromTo(elementsSelector_${sId}, { rotation: 0, opacity: 0 }, { rotation: 360, opacity: 1, transformOrigin: '50% 50%', stagger: 0.1, duration: 1.0 });
        } else if (action_${sId} === 'scale_up') {
          sceneTl.fromTo(elementsSelector_${sId}, { scale: 0, opacity: 0 }, { scale: 1, opacity: 1, transformOrigin: '50% 50%', stagger: 0.1, duration: 0.6, ease: 'back.out(1.5)' });
        } else if (action_${sId} === 'slide_left') {
          sceneTl.fromTo(elementsSelector_${sId}, { x: 150, opacity: 0 }, { x: 0, opacity: 1, stagger: 0.1, duration: 0.6 });
        } else if (action_${sId} === 'slide_right') {
          sceneTl.fromTo(elementsSelector_${sId}, { x: -150, opacity: 0 }, { x: 0, opacity: 1, stagger: 0.1, duration: 0.6 });
        } else {
          sceneTl.fromTo(elementsSelector_${sId}, { opacity: 0 }, { opacity: 1, stagger: 0.1, duration: 0.5 });
        }

        // Per-element-type continuous behavior AFTER the entrance, instead of
        // every element doing the exact same one-shot motion and then
        // freezing. This is what the template spec has always asked the LLM
        // for (dash_array "for flowing/moving lines like rising vapor or
        // arrows") but the renderer never actually implemented until now.
        elementsMeta_${sId}.forEach((el, idx) => {
          const target = document.getElementById('el_comp_${sId}_' + idx);
          if (!target) return;
          const settleTime = 1.0 + idx * 0.1;

          if (el.dash_array) {
            // Flow elements (rising vapor, wind, current, arrows) actually
            // flow continuously - a moving dash pattern - instead of just
            // fading in and sitting still like every other shape.
            const dashLen = (String(el.dash_array).split(/\\s+/).map(Number).reduce((a,b)=>a+b,0)) * 4 || 40;
            gsap.set(target, { strokeDasharray: el.dash_array });
            sceneTl.fromTo(target, { strokeDashoffset: 0 }, { strokeDashoffset: -dashLen, duration: 1.4, repeat: 240, ease: 'none' }, settleTime);
          } else if (el.type === 'circle') {
            // Small circular accents (sun, highlight dots, droplets) get a
            // slow ambient pulse so the scene keeps a hint of motion.
            sceneTl.to(target, { scale: 1.08, transformOrigin: '50% 50%', duration: 1.3, yoyo: true, repeat: 240, ease: 'sine.inOut' }, settleTime);
          }
          // Outline/path shapes with no dash_array (the main drawn form -
          // e.g. the sea, a cell wall, a mountain outline) intentionally get
          // no further motion after settling - a scene where literally
          // everything is in constant motion reads as noisy, not alive.
        });
      }
    `;
  }
};
