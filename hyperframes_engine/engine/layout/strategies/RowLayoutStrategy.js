const LayoutStrategy = require('./LayoutStrategy');

/**
 * RowLayoutStrategy.js
 * Arranges components in a horizontal linear row layout.
 */
class RowLayoutStrategy extends LayoutStrategy {
  constructor() {
    super('ROW');
  }

  calculate(components, parentBounds, layoutConfig) {
    const spacing = layoutConfig.spacing !== undefined ? layoutConfig.spacing : 20;
    const padding = layoutConfig.padding || { top: 0, left: 0, bottom: 0, right: 0 };
    const safeArea = layoutConfig.safeArea || { top: 40, left: 60, bottom: 40, right: 60 };

    let currentX = parentBounds.x + padding.left + safeArea.left;

    components.forEach((comp) => {
      const width = comp.properties.width || 200;
      const height = comp.properties.height || 60;

      const x = currentX;
      const y = parentBounds.y + padding.top + safeArea.top;

      // Assign bounds directly to component style properties
      comp.style.position = 'absolute';
      comp.style.left = `${x}px`;
      comp.style.top = `${y}px`;
      comp.style.width = `${width}px`;
      comp.style.height = `${height}px`;

      currentX += width + spacing;
    });

    return components;
  }
}

module.exports = RowLayoutStrategy;
