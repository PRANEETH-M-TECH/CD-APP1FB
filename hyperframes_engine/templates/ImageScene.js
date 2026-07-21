const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * ImageScene.js
 * Template orchestrator delegating layout rendering entirely to the engine Renderer.
 */
module.exports = {
  render: (sId, data, storyboard) => {
    const sceneJson = storyboard.scenes.find(s => s.scene_no === sId);
    const scene = Scene.deserialize(sceneJson);
    return Renderer.renderScene(scene);
  },
  animate: (sId, data, theme, sceneDuration) => {
    const animStyle = data.animation_style || 'simple_zoom';
    const zoomTargets = data.zoom_targets || [];
    
    let zoomLogic = '';
    if (animStyle === 'simple_zoom') {
      zoomLogic = `sceneTl.to('#img-el-${sId}', { scale: 1.15, duration: ${sceneDuration || 5.0}, ease: 'none' }, 0);`;
    } else if (zoomTargets.length > 0) {
      zoomTargets.forEach(target => {
        const timeAt = (target.at_percent / 100) * (sceneDuration || 5.0);
        zoomLogic += `
          sceneTl.to('#img-el-${sId}', {
            scale: ${target.scale || 1.0},
            x: ${(50 - target.x) * 5} + 'px',
            y: ${(50 - target.y) * 5} + 'px',
            duration: 1.0,
            ease: 'power2.inOut'
          }, ${timeAt});
        `;
      });
    }
    
    return `
      sceneTl.fromTo('#img-el-${sId}', { opacity: 0, scale: 0.8 }, { opacity: 1, scale: 1, duration: 0.6, ease: 'power3.out' });
      ${zoomLogic}
    `;
  }
};
