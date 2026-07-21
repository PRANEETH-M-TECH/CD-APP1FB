const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * TaxonomyTree.js
 * Template renderer and animator for hierarchical taxonomy / classification trees.
 */
module.exports = {
  render: (sId, data, storyboard) => {
    const title = data.title || 'Classification Hierarchy';
    const root = data.root || { label: 'Root Category', children: [{ label: 'Sub-Category A' }, { label: 'Sub-Category B' }] };
    const branches = root.children || [];

    return `
      <div class="taxonomy-container" id="taxonomy-${sId}" style="width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px; position: relative;">
        <h2 class="theme-text" style="font-size: 34px; font-weight: 900; margin-bottom: 30px;" id="taxonomy-title-${sId}">${title}</h2>
        
        <div class="tree-canvas" style="position: relative; width: 90%; height: 450px; display: flex; flex-direction: column; align-items: center;">
          <!-- Root Node -->
          <div class="root-node theme-accent-bg" id="tax-root-${sId}" style="z-index: 5; padding: 16px 36px; border-radius: 20px; font-size: 24px; font-weight: 800; color: #090d16; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
            ${root.label || root}
          </div>

          <!-- Branches Layer -->
          <div id="tax-branches-${sId}" style="display: flex; justify-content: space-around; width: 100%; margin-top: 80px; z-index: 5;">
            ${branches.map((b, bIdx) => `
              <div class="branch-card theme-card-bg theme-card-border" id="tax-branch-${sId}-${bIdx}" style="padding: 18px 24px; border-radius: 16px; min-width: 180px; text-align: center; font-weight: 700; font-size: 18px; box-shadow: 0 8px 20px rgba(0,0,0,0.4);">
                ${b.label || b}
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `;
  },

  animate: (sId, data) => {
    return `
      sceneTl.fromTo('#taxonomy-title-${sId}', { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.4 });
      sceneTl.fromTo('#tax-root-${sId}', { scale: 0, opacity: 0 }, { scale: 1, opacity: 1, duration: 0.5, ease: 'back.out(1.5)' }, 0.2);
      sceneTl.fromTo('#tax-branches-${sId} .branch-card', { opacity: 0, y: 30, scale: 0.8 }, { opacity: 1, y: 0, scale: 1, stagger: 0.15, duration: 0.5, ease: 'power2.out' }, 0.5);
    `;
  }
};
