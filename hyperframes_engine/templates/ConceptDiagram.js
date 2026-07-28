const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * ConceptDiagram.js
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
    const hasBullets = data.left_bullets && data.left_bullets.length > 0;
    const centerX = hasBullets ? 900 : 640;
    const centerY = 360;
    const radius = hasBullets ? 190 : 270;
    const leafNodes = data.leaf_nodes || [];
    const totalLeaves = leafNodes.length;
    const timeStep = (dur - 1.5) / Math.max(1, totalLeaves);
    
    return `
      sceneTl.fromTo('#cd-center-${sId}', { scale: 0 }, { scale: 1, duration: 0.6, ease: 'back.out(1.7)' }, 0.2);
      
      ${hasBullets ? `
        sceneTl.fromTo('#cd-left-title-${sId}', { opacity: 0, x: -20 }, { opacity: 1, x: 0, duration: 0.4 }, 0.3);
        sceneTl.fromTo('#cd-bullets-list-${sId} .bullet-card', { opacity: 0, y: 15 }, { opacity: 1, y: 0, stagger: 0.2, duration: 0.5 }, 0.4);
      ` : ''}

      const linesGroup_${sId} = document.getElementById('cd-lines-group-${sId}');
      const leafNodesData_${sId} = ${JSON.stringify(leafNodes)};
      const dur_${sId} = ${dur};
      const timeStep_${sId} = ${timeStep};
      
      leafNodesData_${sId}.forEach((node, idx) => {
        const total = leafNodesData_${sId}.length;
        let angle = 0;
        if (${hasBullets}) {
          angle = -Math.PI / 3 + (idx * (2 * Math.PI / 3)) / Math.max(1, total - 1);
        } else {
          const leftNodes = [];
          const rightNodes = [];
          for (let i = 0; i < total; i++) {
            if (i % 2 === 0) rightNodes.push(i);
            else leftNodes.push(i);
          }
          if (idx % 2 === 0) {
            const k = rightNodes.indexOf(idx);
            const M = rightNodes.length;
            angle = -Math.PI / 4.5 + (k * (2 * Math.PI / 4.5)) / Math.max(1, M - 1);
          } else {
            const k = leftNodes.indexOf(idx);
            const L = leftNodes.length;
            angle = Math.PI - Math.PI / 4.5 + (k * (2 * Math.PI / 4.5)) / Math.max(1, L - 1);
          }
        }
        
        const x2 = ${centerX} + ${radius} * Math.cos(angle);
        const y2 = ${centerY} + ${radius} * Math.sin(angle);
        
        const lineEl = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        lineEl.setAttribute('x1', ${centerX});
        lineEl.setAttribute('y1', ${centerY});
        lineEl.setAttribute('x2', ${centerX});
        lineEl.setAttribute('y2', ${centerY});
        lineEl.setAttribute('stroke', theme.accentColor);
        lineEl.setAttribute('stroke-width', '3.5');
        lineEl.setAttribute('stroke-linecap', 'round');
        lineEl.style.filter = 'drop-shadow(0 0 6px ' + theme.accentColor + ')';
        lineEl.style.opacity = '0.85';
        linesGroup_${sId}.appendChild(lineEl);

        const leafNode = document.getElementById('cd-leaf-${sId}-' + idx);
        if (leafNode) {
          leafNode.style.left = x2 + 'px';
          leafNode.style.top = y2 + 'px';
          leafNode.style.transform = 'translate(-50%, -50%) scale(0)';
        }

        const revealStart = 0.5 + (idx * timeStep_${sId});

        sceneTl.to(lineEl, {
          attr: { x2: x2, y2: y2 },
          duration: 0.6,
          ease: 'power2.out'
        }, revealStart);

        if (leafNode) {
          sceneTl.to(leafNode, {
            transform: 'translate(-50%, -50%) scale(1)',
            duration: 0.5,
            ease: 'back.out(1.7)'
          }, revealStart + 0.2);

          // Active node pulse glow
          sceneTl.to(leafNode, {
            boxShadow: '0 0 24px ' + theme.accentColor,
            duration: 0.4,
            yoyo: true,
            repeat: 1
          }, revealStart + 0.6);
        }
      });
    `;
  }
};
