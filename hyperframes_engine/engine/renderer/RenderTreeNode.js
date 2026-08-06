/**
 * RenderTreeNode.js
 * Represents a single node inside the renderer-specific Render Tree.
 * Converts pure Component model properties into standard browser HTML/SVG elements.
 */
class RenderTreeNode {
  constructor(node, component, svgContext = false) {
    this.node = node;
    this.component = component;
    this.svgContext = svgContext;
    this.children = [];
  }

  /**
   * Appends a child RenderTreeNode.
   * @param {RenderTreeNode} child 
   */
  addChild(child) {
    this.children.push(child);
  }

  /**
   * Builds a style attribute excluding fill/stroke paint so SVG presentation
   * attributes on the element are not overridden by CSS.
   * @param {object} style
   * @returns {string}
   */
  static _styleWithoutPaint(style) {
    if (!style) return '';
    const skip = new Set(['fill', 'stroke', 'strokeWidth', 'stroke-width', 'color']);
    const entries = Object.keys(style)
      .filter(k => !skip.has(k) && style[k] != null)
      .map(k => {
        const cssKey = k.replace(/([A-Z])/g, '-$1').toLowerCase();
        return `${cssKey}: ${style[k]}`;
      });
    return entries.length > 0 ? `style="${entries.join('; ')}"` : '';
  }

  /**
   * Renders the node and its children recursively into HTML/SVG.
   * @returns {string}
   */
  render() {
    if (!this.component || !this.component.visibility) {
      return '';
    }

    const { type, properties, style } = this.component;
    const childrenHTML = this.children.map(c => c.render()).join('');

    // Generate style attribute if styling exists
    let styleAttr = '';
    const styleKeys = Object.keys(style || {});
    if (styleKeys.length > 0) {
      styleAttr = `style="${styleKeys.map(k => {
        const cssKey = k.replace(/([A-Z])/g, '-$1').toLowerCase();
        return `${cssKey}: ${style[k]}`;
      }).join('; ')}"`;
    }

    switch (type.toUpperCase()) {
      case 'TEXT': {
        if (this.svgContext) {
          const pos = properties.position || {};
          const x = properties.x != null ? properties.x : (pos.x != null ? pos.x : (properties.cx || 0));
          const y = properties.y != null ? properties.y : (pos.y != null ? pos.y : (properties.cy || 0));
          const fill = properties.fill || (style && style.color) || '#ffffff';
          return `<text x="${x}" y="${y}" fill="${fill}" id="${this.component.id}" ${styleAttr}>${properties.text || ''}</text>`;
        }
        return `<span id="${this.component.id}" ${styleAttr}>${properties.text || ''}</span>`;
      }

      case 'IMAGE': {
        return `<img src="${properties.url || ''}" id="${this.component.id}" ${styleAttr} />`;
      }

      case 'SVG': {
        const viewBox = properties.viewBox || '0 0 1280 720';
        // Ensure SVG has explicit pixel size so it does not collapse to 0×0
        const hasSize = styleAttr.includes('width') || styleAttr.includes('height');
        const sizeAttr = hasSize
          ? ''
          : 'width="1280" height="720" style="position: absolute; width: 1280px; height: 720px; top: 0; left: 0;"';
        if (hasSize) {
          return `<svg viewBox="${viewBox}" width="1280" height="720" id="${this.component.id}" ${styleAttr}>${childrenHTML}</svg>`;
        }
        return `<svg viewBox="${viewBox}" width="1280" height="720" id="${this.component.id}" ${sizeAttr}>${childrenHTML}</svg>`;
      }

      case 'SHAPE': {
        // Prefer explicit shapeType; otherwise infer from type / path_data so LLM payloads render correctly.
        // Also override the ComponentFactory default 'rect' when path/circle data is present.
        let shape = (properties.shapeType || properties.type || '').toLowerCase();
        if (properties.path_data || properties.d) {
          shape = 'path';
        } else if (!shape || shape === 'shape' || (shape === 'rect' && properties.r != null)) {
          if (properties.r != null || (properties.cx != null && properties.cy != null)) shape = 'circle';
          else if (properties.x1 != null || properties.y1 != null) shape = 'line';
          else if (!shape || shape === 'shape') shape = 'rect';
        }

        const fill = properties.fill != null ? properties.fill : 'none';
        const stroke = properties.stroke || properties.stroke_color || '#ffffff';
        const strokeWidth = properties.strokeWidth != null
          ? properties.strokeWidth
          : (properties.stroke_width != null ? properties.stroke_width : 2);
        // Avoid CSS fill/stroke overriding SVG presentation attributes
        const shapeStyleAttr = RenderTreeNode._styleWithoutPaint(style);
        const common = `id="${this.component.id}" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}" ${shapeStyleAttr}`;

        // Shape elements (circle/rect/line/path) are SVG leaf tags that cannot
        // visually contain children of their own - a nested <text> label would
        // be silently dropped if returned as-is. When the shape has children
        // (e.g. the LABEL node illustrated_scene attaches for el.label), wrap
        // the shape and its children in an (unlabelled) <g> so the label
        // actually renders as a sibling <text>, while the shape itself keeps
        // its own id unchanged so existing GSAP selectors (document.getElementById
        // targeting el_comp_<scene>_<idx> directly) keep working untouched.
        let shapeHtml;
        if (shape === 'circle' || shape === 'ellipse') {
          shapeHtml = `<circle cx="${properties.cx || 0}" cy="${properties.cy || 0}" r="${properties.r || 0}" ${common} />`;
        } else if (shape === 'rect') {
          shapeHtml = `<rect x="${properties.x || 0}" y="${properties.y || 0}" width="${properties.width || 0}" height="${properties.height || 0}" rx="${properties.rx || 0}" ${common} />`;
        } else if (shape === 'line') {
          shapeHtml = `<line x1="${properties.x1 || 0}" y1="${properties.y1 || 0}" x2="${properties.x2 || 0}" y2="${properties.y2 || 0}" ${common} />`;
        } else if (shape === 'path') {
          const dash = properties.dash_array || properties.strokeDasharray || '';
          shapeHtml = `<path d="${properties.d || properties.path_data || ''}" stroke-dasharray="${dash}" ${common} />`;
        } else {
          return childrenHTML;
        }
        return childrenHTML ? `<g>${shapeHtml}${childrenHTML}</g>` : shapeHtml;
      }

      case 'LABEL': {
        const pos = properties.position || {};
        const x = properties.x != null ? properties.x : (pos.x != null ? pos.x : 0);
        const y = properties.y != null ? properties.y : (pos.y != null ? pos.y : 0);
        const fill = properties.fill || (style && style.color) || '#ffffff';
        return `<text x="${x}" y="${y}" fill="${fill}" id="${this.component.id}" ${styleAttr}>${properties.text || ''}</text>`;
      }

      case 'GROUP': {
        if (this.svgContext) {
          return `<g id="${this.component.id}" ${styleAttr}>${childrenHTML}</g>`;
        }
        return `<div id="${this.component.id}" class="group-container" ${styleAttr}>${childrenHTML}</div>`;
      }

      case 'CUSTOM':
      default: {
        return childrenHTML;
      }
    }
  }
}

module.exports = RenderTreeNode;
