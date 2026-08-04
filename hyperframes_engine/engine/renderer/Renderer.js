const RenderTree = require('./RenderTree');
const { getIconMarkup } = require('../../shared/icons.js');

/**
 * Renderer.js
 * The single rendering authority in the HyperFrames modular engine.
 * Renders structured Scene Graph containers into visual HTML/SVG elements.
 */
class Renderer {
  /**
   * Renders a small inline icon <svg> for a node's optional `icon` name
   * (looked up in the shared curated icon library, falling back to a plain
   * dot for missing/unrecognized names - never breaks rendering). Returns
   * '' when no icon name is given, so callers can splice this in unconditionally.
   * @param {string} iconName
   * @param {number} sizePx
   * @returns {string} HTML for the icon, or '' if iconName is falsy
   */
  static renderIcon(iconName, sizePx = 22) {
    if (!iconName) return '';
    const markup = getIconMarkup(iconName);
    return `<svg viewBox="0 0 24 24" width="${sizePx}" height="${sizePx}" fill="none" class="theme-stroke" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0; display:block; margin: 0 auto 6px;">${markup}</svg>`;
  }

  /**
   * Returns an inline CSS string applying scene.focuses emphasis (dim/highlight/glow/
   * isolate) directly to a rendered element, bypassing the VisualEmphasisEngine/
   * component.style indirection that most template renderers never read. Matches a
   * focus target against either the scene-graph node id or the element's own display
   * text (case-insensitive), since the LLM authoring a storyboard has no visibility
   * into generated node-id naming and more naturally refers to items by their label.
   * @param {Scene} scene
   * @param {string[]} candidateIds - node id(s) and/or display text for this element
   * @returns {string} extra inline CSS (may be empty)
   */
  static getFocusStyle(scene, candidateIds) {
    if (!scene || !Array.isArray(scene.focuses) || scene.focuses.length === 0) return '';
    const mode = (scene.focuses[0].mode || '').toUpperCase();
    const targets = scene.focuses.map(f => String(f.target || '').toLowerCase().trim()).filter(Boolean);
    if (targets.length === 0) return '';
    const isFocused = candidateIds.some(cid => targets.includes(String(cid || '').toLowerCase().trim()));

    switch (mode) {
      case 'DIM_BACKGROUND':
        return isFocused ? '' : 'opacity:0.35; filter: blur(1px) grayscale(40%); transition: opacity 0.4s ease, filter 0.4s ease;';
      case 'HIGHLIGHT':
        return isFocused ? 'filter: drop-shadow(0 0 12px #3b82f6); transition: filter 0.4s ease;' : '';
      case 'GLOW':
        return isFocused ? 'filter: drop-shadow(0 0 20px #eab308); transition: filter 0.4s ease;' : '';
      case 'ISOLATE':
        return isFocused ? '' : 'visibility:hidden;';
      default:
        return '';
    }
  }

  /**
   * Main entry method to render a Scene.
   * @param {Scene} scene
   * @returns {string} Compiled HTML markup
   */
  static renderScene(scene) {
    if (!scene) {
      throw new Error("[Renderer Error] Cannot render null or undefined scene.");
    }

    const templateId = scene.templateId || 'general_scene';
    
    // Structured developer logging
    console.log(`[Renderer LOG] Processing rendering pipeline for Scene ${scene.sceneNo} (${templateId})`);

    // Run structural validations
    Renderer.validateScene(scene);

    // Execute Layout Manager positioning calculations
    const LayoutManager = require('../layout/managers/LayoutManager');
    LayoutManager.layoutScene(scene);

    // Execute Focus & Layer Engine calculations
    const LayerManager = require('../focus/layers/LayerManager');
    const AttentionManager = require('../focus/manager/AttentionManager');
    const VisualEmphasisEngine = require('../focus/effects/VisualEmphasisEngine');

    LayerManager.applyLayering(scene);
    AttentionManager.resolveSceneFocus(scene);
    VisualEmphasisEngine.applyEmphasis(scene);

    // Build the internal Render Tree representation
    const renderTree = RenderTree.build(scene);

    // Resolve Camera transform style
    const Camera = require('../camera/models/Camera');
    const CameraController = require('../camera/controllers/CameraController');
    const camera = scene.camera || new Camera();
    const cameraController = new CameraController(camera);
    const cameraTransform = cameraController.getTransformStyle();

    // Render using specific template frame structures for backward-compatible visuals
    let contentHtml = '';
    switch (templateId) {
      case 'title_slide':
        contentHtml = Renderer.renderTitleSlide(scene, renderTree);
        break;
      case 'concept_diagram':
        contentHtml = Renderer.renderConceptDiagram(scene, renderTree);
        break;
      case 'cycle_template':
        contentHtml = Renderer.renderCycleTemplate(scene, renderTree);
        break;
      case 'math_derivation':
        contentHtml = Renderer.renderMathDerivation(scene, renderTree);
        break;
      case 'column_comparison':
        contentHtml = Renderer.renderColumnComparison(scene, renderTree);
        break;
      case 'horizontal_timeline':
        contentHtml = Renderer.renderHorizontalTimeline(scene, renderTree);
        break;
      case 'database_grid':
        contentHtml = Renderer.renderDatabaseGrid(scene, renderTree);
        break;
      case 'taxonomy_tree':
        contentHtml = Renderer.renderTaxonomyTree(scene, renderTree);
        break;
      case 'cartesian_grid':
        contentHtml = Renderer.renderCartesianGrid(scene, renderTree);
        break;
      case 'geo_marker':
        contentHtml = Renderer.renderGeoMarker(scene, renderTree);
        break;
      case 'before_after_slider':
        contentHtml = Renderer.renderBeforeAfterSlider(scene, renderTree);
        break;
      case 'venn_diagram':
        contentHtml = Renderer.renderVennDiagram(scene, renderTree);
        break;
      case 'quiz_checkpoint':
        contentHtml = Renderer.renderQuizCheckpoint(scene, renderTree);
        break;
      case 'illustrated_scene':
        contentHtml = Renderer.renderIllustratedScene(scene, renderTree);
        break;
      case 'image_scene':
        contentHtml = Renderer.renderImageScene(scene, renderTree);
        break;
      case 'general_scene':
      default:
        contentHtml = Renderer.renderGeneralScene(scene, renderTree);
        break;
    }

    // Wrap the content inside the camera viewport container
    const wrapped = `
      <div class="camera-viewport-wrapper" style="width: 100%; height: 100%; position: relative; overflow: hidden; ${cameraTransform}">
        ${contentHtml}
      </div>
    `;

    // If contentHtml is empty or minimal, log a warning for template generation issues
    if (!contentHtml || contentHtml.trim().length < 10) {
      console.warn(`[Renderer Warning] Rendered empty content for Scene ${scene.sceneNo} (template: ${templateId})`);
    }

    return wrapped;
  }

  /**
   * Asserts structural validations on the scene graph to prevent engine failures.
   */
  static validateScene(scene) {
    if (!scene.nodes || !Array.isArray(scene.nodes)) {
      throw new Error(`[Renderer Validation Error] Scene ${scene.sceneNo} has invalid or missing node array.`);
    }

    scene.traverse((node) => {
      if (!node) {
        throw new Error(`[Renderer Validation Error] Scene ${scene.sceneNo} contains a null node.`);
      }
      if (!node.component) {
        throw new Error(`[Renderer Validation Error] Node '${node.id}' is missing a component instance.`);
      }
      // Check for valid component structure
      const required = ['id', 'type', 'properties', 'style', 'children'];
      for (const req of required) {
        if (node.component[req] === undefined) {
          throw new Error(`[Renderer Validation Error] Component '${node.component.id}' is missing required field '${req}'.`);
        }
      }
    });
  }

  /* ====================================================
     TEMPLATE LAYOUT RENDERERS (BACKWARD COMPATIBLE)
     ==================================================== */

  static renderTitleSlide(scene, renderTree) {
    const sId = scene.sceneNo;
    const titleNode = scene.findNode(`title_${sId}`);
    const subtitleNode = scene.findNode(`subtitle_${sId}`);

    const title = titleNode ? titleNode.component.properties.text : '';
    const subtitle = subtitleNode ? subtitleNode.component.properties.text : '';

    return `
      <div class="title-slide-container" id="title-slide-${sId}">
        <div class="icon-card" id="icon-card-${sId}">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="theme-stroke">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" id="icon-path-${sId}" />
          </svg>
        </div>
        <h1 class="theme-text" id="title-text-${sId}" style="font-size: 56px; font-weight: 900; margin-bottom: 16px; text-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);">${title}</h1>
        <p id="subtitle-text-${sId}" style="font-size: 22px; font-weight: 500; color: rgba(255, 255, 255, 0.7); text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3); max-width: 850px; text-align: center; line-height: 1.4;">${subtitle}</p>
      </div>
    `;
  }

  static renderConceptDiagram(scene, renderTree) {
    const sId = scene.sceneNo;
    const centerNode = scene.findNode(`center_${sId}`);
    const leavesNode = scene.findNode(`leaves_${sId}`);
    const leftTitleNode = scene.findNode(`left_title_${sId}`);
    const bulletsNode = scene.findNode(`bullets_${sId}`);

    const hasBullets = !!bulletsNode;
    const leftTitle = leftTitleNode ? leftTitleNode.component.properties.text : '';
    const bullets = bulletsNode ? bulletsNode.component.children.map(c => ({ text: c.properties.text, icon: c.properties.icon })) : [];
    const centerText = centerNode ? centerNode.component.properties.text : '';
    const centerIcon = centerNode ? centerNode.component.properties.icon : null;
    const leaves = leavesNode ? leavesNode.component.children.map(c => ({ text: c.properties.text, icon: c.properties.icon })) : [];

    return `
      <div class="concept-diagram-container" id="concept-diagram-${sId}">
        ${hasBullets ? `
        <div class="left-bullets-col">
          <h2 class="theme-text" id="cd-left-title-${sId}" style="font-size: 38px; font-weight: 800; margin-bottom: 24px; letter-spacing: -1px;">${leftTitle}</h2>
          <div style="display: flex; flex-direction: column; gap: 16px;" id="cd-bullets-list-${sId}">
            ${bullets.map((b, bIdx) => `<div class="bullet-card theme-card-bg theme-card-border" id="cd-bullet-${sId}-${bIdx}" style="display:flex; align-items:center; gap:12px; ${Renderer.getFocusStyle(scene, [`bullet_${sId}_${bIdx}`, b.text])}">${Renderer.renderIcon(b.icon, 24).replace('margin: 0 auto 6px;', 'margin:0;')}<span>${b.text}</span></div>`).join('')}
          </div>
        </div>
        ` : ''}

        <div class="mindmap-canvas" style="position: relative; width: ${hasBullets ? '55%' : '100%'}; height: 100%;">
          <svg viewBox="0 0 1280 720" style="position: absolute; width: 1280px; height: 720px; top: 0; left: 0; z-index: 2; pointer-events: none;">
            <g id="cd-lines-group-${sId}"></g>
          </svg>
          <div class="center-node theme-accent-bg" id="cd-center-${sId}" style="color: #090d16; position: absolute; left: ${hasBullets ? 900 : 640}px; top: 360px; transform: translate(-50%, -50%); ${centerIcon ? 'flex-direction: column;' : ''} ${Renderer.getFocusStyle(scene, [`center_${sId}`, centerText])}">${Renderer.renderIcon(centerIcon, 26)}<span>${centerText}</span></div>
          <div id="cd-leaves-group-${sId}">
            ${leaves.map((leaf, nIdx) => `<div class="leaf-node theme-card-border" id="cd-leaf-${sId}-${nIdx}" style="position: absolute; transform: translate(-50%, -50%) scale(0); ${Renderer.getFocusStyle(scene, [`leaf_${sId}_${nIdx}`, leaf.text])}">${Renderer.renderIcon(leaf.icon, 20)}<span>${leaf.text}</span></div>`).join('')}
          </div>
        </div>
      </div>
    `;
  }

  static renderCycleTemplate(scene, renderTree) {
    const sId = scene.sceneNo;
    const titleNode = scene.findNode(`title_${sId}`);
    const stagesNode = scene.findNode(`stages_${sId}`);

    const title = titleNode ? titleNode.component.properties.text : '';
    const stages = stagesNode ? stagesNode.component.children.map(c => ({ text: c.properties.text, icon: c.properties.icon })) : [];

    return `
      <div class="cycle-container" id="cycle-${sId}">
        <h2 class="theme-text" style="font-size: 36px; font-weight: 800; text-align: center; margin-bottom: 30px;" id="cycle-title-${sId}">${title}</h2>
        <div class="cycle-canvas">
          <svg class="cycle-svg">
            <circle cx="200" cy="200" r="140" fill="none" stroke-width="4" id="cycle-circle-path-${sId}" class="theme-stroke" style="opacity: 0.35;"></circle>
            <circle cx="200" cy="200" r="7" fill="white" id="cycle-orbit-dot-${sId}" class="theme-fill"></circle>
          </svg>
          <div id="cycle-stages-${sId}">
            ${stages.map((stage, stIdx) => `
              <div class="cycle-stage theme-card-bg theme-card-border" id="cycle-stage-${sId}-${stIdx}">
                <div class="cycle-stage-badge theme-text">Step ${stIdx + 1}</div>
                ${Renderer.renderIcon(stage.icon, 22)}
                <div class="cycle-stage-label">${stage.text}</div>
              </div>
            `).join('')}
          </div>
          <div style="position: absolute; font-size: 12px; text-transform: uppercase; font-weight: 800; letter-spacing: 1px; color: rgba(255, 255, 255, 0.4);" id="cycle-center-label-${sId}">🔄 Cycle Flow</div>
        </div>
      </div>
    `;
  }

  static renderMathDerivation(scene, renderTree) {
    const sId = scene.sceneNo;
    const titleNode = scene.findNode(`title_${sId}`);
    const formulaNode = scene.findNode(`formula_${sId}`);
    const stepsNode = scene.findNode(`steps_${sId}`);

    const title = titleNode ? titleNode.component.properties.text : '';
    const steps = stepsNode ? stepsNode.component.children.map(c => c.properties.text) : [];
    const hasFormula = !!formulaNode;

    return `
      <div class="math-container" id="math-${sId}">
        <h2 class="theme-text" style="font-size: 34px; font-weight: 800; margin-bottom: 24px;" id="math-title-${sId}">${title}</h2>
        ${hasFormula ? `<div class="math-formula-board theme-card-border" id="math-formula-${sId}"></div>` : ''}
        <div style="width: 85%; display: flex; flex-direction: column; gap: 12px;" id="math-steps-${sId}">
          ${steps.map((step, stIdx) => `
            <div class="math-step-card theme-card-bg theme-card-border" id="math-step-${sId}-${stIdx}">
              <div class="math-step-badge theme-accent-bg" id="math-badge-${sId}-${stIdx}">${stIdx + 1}</div>
              <div class="math-step-text" id="math-step-text-${sId}-${stIdx}">${step}</div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  static renderColumnComparison(scene, renderTree) {
    const sId = scene.sceneNo;
    const titleNode = scene.findNode(`title_${sId}`);
    const leftColNode = scene.findNode(`left_col_${sId}`);
    const rightColNode = scene.findNode(`right_col_${sId}`);

    const title = titleNode ? titleNode.component.properties.text : 'Comparison';
    
    const leftHeader = leftColNode ? leftColNode.component.properties.header : '';
    const leftBullets = leftColNode ? leftColNode.component.children.map(c => ({ text: c.properties.text, icon: c.properties.icon })) : [];

    const rightHeader = rightColNode ? rightColNode.component.properties.header : '';
    const rightBullets = rightColNode ? rightColNode.component.children.map(c => ({ text: c.properties.text, icon: c.properties.icon })) : [];

    // Shrink bullet typography as list length grows so tall columns don't
    // overflow the fixed 720px canvas height (body has overflow:hidden, so
    // overflow was previously silently clipped rather than scrolling).
    const maxBullets = Math.max(leftBullets.length, rightBullets.length);
    let bulletFontSize = 18, bulletMarginBottom = 16;
    if (maxBullets > 7) {
      bulletFontSize = 14; bulletMarginBottom = 8;
    } else if (maxBullets > 5) {
      bulletFontSize = 16; bulletMarginBottom = 12;
    }
    const bulletStyle = `font-size: ${bulletFontSize}px; margin-bottom: ${bulletMarginBottom}px;`;

    return `
      <div class="comparison-container" id="comparison-${sId}">
        <h2 class="theme-text" style="font-size: 34px; font-weight: 800; margin-bottom: 24px;" id="comparison-title-${sId}">${title}</h2>
        <div class="comparison-grid">
          <div class="comparison-column theme-card-bg theme-card-border" id="comparison-col-left-${sId}">
            <div class="comparison-col-header theme-text theme-card-border" id="comparison-header-left-${sId}" style="border-bottom-color: rgba(255,255,255,0.1);">${leftHeader}</div>
            <div id="comparison-bullets-left-${sId}">
              ${leftBullets.map((b, bIdx) => `
                <div class="comparison-bullet" id="comparison-bullet-l-${sId}-${bIdx}" style="${bulletStyle} ${Renderer.getFocusStyle(scene, [`left_bullet_${sId}_${bIdx}`, b.text])}">
                  ${b.icon ? Renderer.renderIcon(b.icon, 18).replace('margin: 0 auto 6px;', 'margin:0 10px 0 0;') : '<span class="comparison-bullet-dot theme-accent-bg"></span>'}
                  <span>${b.text}</span>
                </div>
              `).join('')}
            </div>
          </div>

          <div class="comparison-column theme-card-bg theme-card-border" id="comparison-col-right-${sId}">
            <div class="comparison-col-header theme-text theme-card-border" id="comparison-header-right-${sId}" style="border-bottom-color: rgba(255,255,255,0.1);">${rightHeader}</div>
            <div id="comparison-bullets-right-${sId}">
              ${rightBullets.map((b, bIdx) => `
                <div class="comparison-bullet" id="comparison-bullet-r-${sId}-${bIdx}" style="${bulletStyle} ${Renderer.getFocusStyle(scene, [`right_bullet_${sId}_${bIdx}`, b.text])}">
                  ${b.icon ? Renderer.renderIcon(b.icon, 18).replace('margin: 0 auto 6px;', 'margin:0 10px 0 0;') : '<span class="comparison-bullet-dot theme-accent-bg"></span>'}
                  <span>${b.text}</span>
                </div>
              `).join('')}
            </div>
          </div>
        </div>
      </div>
    `;
  }

  static renderHorizontalTimeline(scene, renderTree) {
    const sId = scene.sceneNo;
    const titleNode = scene.findNode(`title_${sId}`);
    const stagesNode = scene.findNode(`stages_${sId}`);

    const title = titleNode ? titleNode.component.properties.text : '';
    const stages = stagesNode ? stagesNode.component.children.map(c => ({
      label: c.properties.text,
      step_no: c.metadata.step_no || 1
    })) : [];

    return `
      <div class="timeline-container" id="timeline-${sId}">
        <h2 style="font-size: 36px; font-weight: 800; color: #ffffff; text-align: center; margin: 0 0 80px 0; text-transform: uppercase; letter-spacing: 1px; text-shadow: 0 4px 8px rgba(0,0,0,0.5);" class="theme-text" id="timeline-title-${sId}">${title}</h2>
        <div class="timeline-track">
          <svg class="timeline-svg">
            <line x1="5%" y1="50%" x2="95%" y2="50%" stroke="rgba(255,255,255,0.06)" stroke-width="4" stroke-linecap="round"></line>
            <line x1="5%" y1="50%" x2="5%" y2="50%" stroke-width="4" stroke-linecap="round" id="timeline-active-line-${sId}" class="theme-stroke"></line>
          </svg>
          
          <div id="timeline-stages-${sId}">
            ${stages.map((stage, stIdx) => `
              <div class="timeline-stage" id="timeline-stage-${sId}-${stIdx}">
                <div class="timeline-stage-circle theme-card-border">
                  <div class="timeline-stage-badge theme-accent-bg">${stage.step_no}</div>
                  <svg viewBox="0 0 24 24" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" class="theme-stroke">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" id="timeline-icon-${sId}-${stIdx}" />
                  </svg>
                </div>
                <p class="timeline-stage-label">${stage.label}</p>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `;
  }

  static renderDatabaseGrid(scene, renderTree) {
    const sId = scene.sceneNo;
    const titleNode = scene.findNode(`title_${sId}`);
    const headersNode = scene.findNode(`headers_${sId}`);
    const rowsNode = scene.findNode(`rows_${sId}`);

    const title = titleNode ? titleNode.component.properties.text : '';
    const headers = headersNode ? headersNode.component.children.map(c => c.properties.text) : [];
    
    // Resolve matrix
    const rows = [];
    if (rowsNode) {
      rowsNode.component.children.forEach((rowComp) => {
        rows.push(rowComp.children.map(c => c.properties.text));
      });
    }

    // Shrink row padding/font-size as row count grows, and cap the card's
    // height with a scroll fallback, so a long table doesn't silently clip
    // against the fixed 720px canvas (body has overflow:hidden).
    let cellPadding = '16px 20px', tableFontSize = 16;
    if (rows.length > 8) {
      cellPadding = '8px 14px'; tableFontSize = 13;
    } else if (rows.length > 5) {
      cellPadding = '11px 16px'; tableFontSize = 14;
    }
    const cellStyle = `padding: ${cellPadding};`;

    return `
      <div class="database-container" id="db-${sId}">
        <h2 class="theme-text" style="font-size: 34px; font-weight: 800; margin-bottom: 24px;" id="db-title-${sId}">${title}</h2>
        <div class="database-grid-card theme-card-border" id="db-card-${sId}" style="max-height: 520px; overflow-y: auto;">
          <table class="database-table" style="font-size: ${tableFontSize}px;">
            <thead>
              <tr id="db-head-row-${sId}">
                ${headers.map((h, hIdx) => `<th id="db-th-${sId}-${hIdx}" style="${cellStyle}">${h}</th>`).join('')}
              </tr>
            </thead>
            <tbody>
              ${rows.map((row, rIdx) => `
                <tr id="db-row-${sId}-${rIdx}">
                  ${row.map((cell, cIdx) => `<td id="db-cell-${sId}-${rIdx}-${cIdx}" style="${cellStyle}">${cell}</td>`).join('')}
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  static renderVennDiagram(scene, renderTree) {
    const sId = scene.sceneNo;
    const leftTitleNode = scene.findNode(`left_title_${sId}`);
    const rightTitleNode = scene.findNode(`right_title_${sId}`);
    
    const leftNode = scene.findNode(`left_${sId}`);
    const midNode = scene.findNode(`intersection_${sId}`);
    const rightNode = scene.findNode(`right_${sId}`);

    const leftTitle = leftTitleNode ? leftTitleNode.component.properties.text : 'A';
    const rightTitle = rightTitleNode ? rightTitleNode.component.properties.text : 'B';

    const leftItems = leftNode ? leftNode.component.children.map(c => ({ text: c.properties.text, icon: c.properties.icon })) : [];
    const midItems = midNode ? midNode.component.children.map(c => ({ text: c.properties.text, icon: c.properties.icon })) : [];
    const rightItems = rightNode ? rightNode.component.children.map(c => ({ text: c.properties.text, icon: c.properties.icon })) : [];
    const itemInner = (item) => item.icon
      ? `${Renderer.renderIcon(item.icon, 16).replace('margin: 0 auto 6px;', 'margin:0 8px 0 0;')}<span>${item.text}</span>`
      : item.text;
    const itemStyle = (item) => item.icon ? 'display:flex; align-items:center;' : '';

    return `
      <div class="venn-container" id="venn-${sId}">
        <div class="venn-headers">
          <div class="theme-text" id="venn-header-left-${sId}">${leftTitle}</div>
          <div style="color: #ffffff;">Comparison</div>
          <div class="theme-text" id="venn-header-right-${sId}">${rightTitle}</div>
        </div>

        <div class="venn-diagram-canvas">
          <div class="venn-circle-left theme-card-bg theme-card-border" id="venn-circle-left-${sId}"></div>
          <div class="venn-circle-right theme-card-bg theme-card-border" id="venn-circle-right-${sId}"></div>

          <div class="venn-content-left" id="venn-content-left-${sId}">
            ${leftItems.map((item, iIdx) => `<div class="venn-item-card" id="venn-item-l-${sId}-${iIdx}" style="${itemStyle(item)}">${itemInner(item)}</div>`).join('')}
          </div>

          <div class="venn-content-middle" id="venn-content-middle-${sId}">
            ${midItems.map((item, iIdx) => `<div class="venn-item-card theme-accent-border" id="venn-item-m-${sId}-${iIdx}" style="border: 1.5px dashed; font-weight: 700; ${itemStyle(item)}">${itemInner(item)}</div>`).join('')}
          </div>

          <div class="venn-content-right" id="venn-content-right-${sId}">
            ${rightItems.map((item, iIdx) => `<div class="venn-item-card" id="venn-item-r-${sId}-${iIdx}" style="${itemStyle(item)}">${itemInner(item)}</div>`).join('')}
          </div>
        </div>
      </div>
    `;
  }

  static renderQuizCheckpoint(scene, renderTree) {
    const sId = scene.sceneNo;
    const questionNode = scene.findNode(`question_${sId}`);
    const optionsNode = scene.findNode(`options_${sId}`);

    const question = questionNode ? questionNode.component.properties.text : '';
    const options = optionsNode ? optionsNode.component.children.map(c => c.properties.text) : [];

    return `
      <div class="quiz-container" id="quiz-${sId}">
        <div class="quiz-card theme-card-border" id="quiz-card-${sId}">
          <div class="quiz-question" id="quiz-question-${sId}">${question}</div>
          <div class="quiz-options-list" id="quiz-options-${sId}">
            ${options.map((optionText, oIdx) => `
              <div class="quiz-option theme-card-bg theme-card-border" id="quiz-opt-${sId}-${oIdx}">
                <div class="quiz-option-index theme-accent-bg" id="quiz-opt-idx-${sId}-${oIdx}">${String.fromCharCode(65 + oIdx)}</div>
                <div class="quiz-option-text">${optionText}</div>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `;
  }

  static renderIllustratedScene(scene, renderTree) {
    const sId = scene.sceneNo;
    const canvasNode = scene.findNode(`canvas_${sId}`);
    const titleNode = scene.findNode(`title_${sId}`);
    const canvasColor = (canvasNode && canvasNode.component.properties.color) || 'transparent';
    const title = titleNode
      ? titleNode.component.properties.text
      : (scene.metadata && scene.metadata.title) || '';

    // Delegate shape/path/circle/label rendering to the Render Tree (SVG context)
    const canvasRtNode = renderTree.rootNodes.find(n => n.node.id === `canvas_${sId}`);
    const svgContent = canvasRtNode ? canvasRtNode.render() : '';

    return `
      <div class="illustrated-canvas" id="ill-canvas-${sId}" style="background-color: ${canvasColor}; width: 100%; height: 100%; position: relative; overflow: hidden;">
        ${title ? `<h2 class="theme-text" id="ill-title-${sId}" style="position: absolute; top: 28px; left: 48px; z-index: 20; font-size: 28px; font-weight: 800; margin: 0; text-shadow: 0 2px 8px rgba(0,0,0,0.5);">${title}</h2>` : ''}
        ${svgContent}
      </div>
    `;
  }

  static renderImageScene(scene, renderTree) {
    const sId = scene.sceneNo;
    const imageNode = scene.findNode(`image_${sId}`);
    const imageUrl = imageNode ? imageNode.component.properties.url : '';

    return `
      <div class="image-scene-container" id="img-scene-${sId}">
        <img src="${imageUrl}" class="scene-image" id="img-el-${sId}" />
        <svg viewBox="0 0 1280 720" style="position: absolute; width: 100%; height: 100%; z-index: 10; pointer-events: none;" id="img-svg-${sId}">
          <g id="img-annotations-${sId}"></g>
        </svg>
      </div>
    `;
  }

  static renderGeneralScene(scene, renderTree) {
    const sId = scene.sceneNo;
    const titleNode = scene.findNode(`title_${sId}`);
    const title = titleNode ? titleNode.component.properties.text : (scene.metadata ? scene.metadata.title : '');

    return `
      <div class="general-scene-container" id="general-${sId}" style="width:100%; height:100%; position:relative;">
        <h2 class="theme-text" style="font-size: 34px; font-weight: 800; position:absolute; top:40px; left:60px;" id="general-title-${sId}">${title}</h2>
        <div id="general-assets-${sId}"></div>
      </div>
    `;
  }

  static renderTaxonomyTree(scene, renderTree) {
    const sId = scene.sceneNo;
    const titleNode = scene.findNode(`title_${sId}`);
    const rootNode = scene.findNode(`root_${sId}`);
    const branchesNode = scene.findNode(`branches_${sId}`);

    const title = titleNode ? titleNode.component.properties.text : 'Classification Hierarchy';
    const rootLabel = rootNode ? rootNode.component.properties.text : 'Root Category';
    const rootIcon = rootNode ? rootNode.component.properties.icon : null;

    // Scale card size/typography down as branch count grows so items wrap onto
    // multiple rows instead of overflowing the fixed 1280x720 canvas (no-wrap
    // flexbox with a fixed min-width previously broke past ~6 branches).
    const branchCount = branchesNode ? branchesNode.component.children.length : 0;
    let cardMinWidth = 180, cardFontSize = 18, cardPadding = '18px 24px', subFontSize = 13;
    if (branchCount > 8) {
      cardMinWidth = 120; cardFontSize = 14; cardPadding = '12px 14px'; subFontSize = 11;
    } else if (branchCount > 6) {
      cardMinWidth = 145; cardFontSize = 16; cardPadding = '14px 18px'; subFontSize = 12;
    }

    let branchesHtml = '';
    if (branchesNode) {
      branchesNode.component.children.forEach((bComp, idx) => {
        const label = bComp.properties.text || '';
        const icon = bComp.properties.icon;
        const sub = bComp.properties.sub || [];

        let subHtml = '';
        if (sub.length > 0) {
          subHtml = `<div style="font-size: ${subFontSize}px; font-weight: 400; opacity: 0.85; line-height: 1.3;">` +
                    sub.map(s => `• ${s}`).join('<br>') +
                    `</div>`;
        }

        branchesHtml += `
          <div class="branch-card theme-card-bg theme-card-border" id="tax-branch-${sId}-${idx}" style="padding: ${cardPadding}; border-radius: 16px; min-width: ${cardMinWidth}px; text-align: center; font-weight: 700; font-size: ${cardFontSize}px; box-shadow: 0 8px 20px rgba(0,0,0,0.4); ${Renderer.getFocusStyle(scene, [`branch_${sId}_${idx}`, label])}">
            ${Renderer.renderIcon(icon, 20)}
            <div style="font-weight: 800; margin-bottom: 4px;">${label}</div>
            ${subHtml}
          </div>
        `;
      });
    }

    return `
      <div class="taxonomy-container" id="taxonomy-${sId}" style="width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px; position: relative;">
        <h2 class="theme-text" style="font-size: 34px; font-weight: 900; margin-bottom: 30px;" id="taxonomy-title-${sId}">${title}</h2>

        <div class="tree-canvas" style="position: relative; width: 90%; height: 450px; display: flex; flex-direction: column; align-items: center; overflow: visible;">
          <!-- Root Node -->
          <div class="root-node theme-accent-bg" id="tax-root-${sId}" style="z-index: 5; padding: 16px 36px; border-radius: 20px; font-size: 24px; font-weight: 800; color: #090d16; box-shadow: 0 10px 25px rgba(0,0,0,0.5); ${rootIcon ? 'display:flex; align-items:center; gap:10px;' : ''}">
            ${Renderer.renderIcon(rootIcon, 22).replace('margin: 0 auto 6px;', 'margin:0;')}<span>${rootLabel}</span>
          </div>

          <!-- Branches Layer -->
          <div id="tax-branches-${sId}" style="display: flex; flex-wrap: wrap; justify-content: center; gap: 16px; width: 100%; margin-top: 80px; z-index: 5;">
            ${branchesHtml}
          </div>
        </div>
      </div>
    `;
  }

  static renderCartesianGrid(scene, renderTree) {
    const sId = scene.sceneNo;
    const titleNode = scene.findNode(`title_${sId}`);
    const equationNode = scene.findNode(`equation_${sId}`);
    const pointsNode = scene.findNode(`points_${sId}`);
    const curveNode = scene.findNode(`curve_${sId}`);

    const title = titleNode ? titleNode.component.properties.text : 'Coordinate Geometry';
    const eqLabel = equationNode ? equationNode.component.properties.text : 'y = f(x)';
    const points = pointsNode ? pointsNode.component.children.map(c => ({
      x: c.properties.x,
      y: c.properties.y,
      label: c.properties.label
    })) : [];
    // Use the LLM's own drawn curve when provided; otherwise fall back to a
    // generic demo curve so the scene never renders with a blank plot area.
    const curveD = curveNode ? curveNode.component.properties.d : 'M 100 300 Q 340 50 580 300';
    const curveStroke = curveNode ? curveNode.component.properties.stroke : '#38bdf8';
    const curveStrokeWidth = curveNode ? curveNode.component.properties.stroke_width : 4;

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
            <path id="cartesian-curve-${sId}" d="${curveD}" stroke="${curveStroke}" stroke-width="${curveStrokeWidth}" fill="none" stroke-linecap="round" />
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
  }

  static renderGeoMarker(scene, renderTree) {
    const sId = scene.sceneNo;
    const titleNode = scene.findNode(`title_${sId}`);
    const markersNode = scene.findNode(`markers_${sId}`);

    const title = titleNode ? titleNode.component.properties.text : 'Geographical Overview';
    const markers = markersNode ? markersNode.component.children.map(c => ({
      label: c.properties.label,
      x: c.properties.x,
      y: c.properties.y,
      description: c.properties.description,
      icon: c.properties.icon
    })) : [];

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
                <div class="pin-card theme-card-bg theme-card-border" style="margin-top: 10px; padding: 10px 16px; border-radius: 12px; white-space: nowrap; text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.6); ${m.icon ? 'display:flex; align-items:center; gap:8px;' : ''}">
                  ${m.icon ? Renderer.renderIcon(m.icon, 18).replace('margin: 0 auto 6px;', 'margin:0;') : ''}
                  <div>
                    <div style="font-weight: 800; font-size: 15px; color: #ffffff;">${m.label || 'Marker'}</div>
                    ${m.description ? `<div style="font-size: 12px; color: rgba(255,255,255,0.7); margin-top: 2px;">${m.description}</div>` : ''}
                  </div>
                </div>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `;
  }

  static renderBeforeAfterSlider(scene, renderTree) {
    const sId = scene.sceneNo;
    const titleNode = scene.findNode(`title_${sId}`);
    const beforeGroup = scene.findNode(`before_group_${sId}`);
    const afterGroup = scene.findNode(`after_group_${sId}`);

    const title = titleNode ? titleNode.component.properties.text : 'Before vs After State';
    const beforeLabel = beforeGroup ? beforeGroup.component.properties.label : 'BEFORE';
    const beforeBullets = beforeGroup ? beforeGroup.component.children.map(c => ({ text: c.properties.text, icon: c.properties.icon })) : [];

    const afterLabel = afterGroup ? afterGroup.component.properties.label : 'AFTER';
    const afterBullets = afterGroup ? afterGroup.component.children.map(c => ({ text: c.properties.text, icon: c.properties.icon })) : [];

    return `
      <div class="before-after-container" id="ba-${sId}" style="width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px; position: relative;">
        <h2 class="theme-text" style="font-size: 34px; font-weight: 900; margin-bottom: 24px;" id="ba-title-${sId}">${title}</h2>

        <div class="ba-grid" style="display: flex; gap: 30px; width: 85%; height: 420px; position: relative; align-items: center; justify-content: center;">
          <!-- Before Card -->
          <div class="ba-card theme-card-bg theme-card-border" id="ba-before-${sId}" style="flex: 1; height: 100%; border-radius: 24px; padding: 30px; display: flex; flex-direction: column; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border-left: 4px solid #ef4444;">
            <div style="font-size: 22px; font-weight: 900; color: #ef4444; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 1px;">${beforeLabel}</div>
            <div style="display: flex; flex-direction: column; gap: 12px;">
              ${beforeBullets.map((b, bIdx) => `
                <div style="font-size: 16px; font-weight: 600; color: rgba(255,255,255,0.9); display: flex; align-items: center; gap: 10px;">
                  ${b.icon ? Renderer.renderIcon(b.icon, 18).replace('margin: 0 auto 6px;', 'margin:0;') : '<span style="width: 8px; height: 8px; border-radius: 50%; background: #ef4444; flex-shrink: 0;"></span>'}
                  <span>${b.text}</span>
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
            <div style="font-size: 22px; font-weight: 900; color: #22c55e; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 1px;">${afterLabel}</div>
            <div style="display: flex; flex-direction: column; gap: 12px;">
              ${afterBullets.map((b, bIdx) => `
                <div style="font-size: 16px; font-weight: 600; color: rgba(255,255,255,0.9); display: flex; align-items: center; gap: 10px;">
                  ${b.icon ? Renderer.renderIcon(b.icon, 18).replace('margin: 0 auto 6px;', 'margin:0;') : '<span style="width: 8px; height: 8px; border-radius: 50%; background: #22c55e; flex-shrink: 0;"></span>'}
                  <span>${b.text}</span>
                </div>
              `).join('')}
            </div>
          </div>
        </div>
      </div>
    `;
  }
}

module.exports = Renderer;
