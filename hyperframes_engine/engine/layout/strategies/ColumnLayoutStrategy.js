const LayoutStrategy = require('./LayoutStrategy');

/**
 * ColumnLayoutStrategy.js
 * Arranges components in a vertical linear stack layout.
 */
class ColumnLayoutStrategy extends LayoutStrategy {
  constructor() {
    super('COLUMN');
  }

  calculate(components, parentBounds, layoutConfig) {
    const spacing = layoutConfig.spacing !== undefined ? layoutConfig.spacing : 20;
    const padding = layoutConfig.padding || { top: 0, left: 0, bottom: 0, right: 0 };
    const safeArea = layoutConfig.safeArea || { top: 40, left: 60, bottom: 40, right: 60 };

    let currentY = parentBounds.y + padding.top + safeArea.top;

    components.forEach((comp) => {
      // Fetch constraints or layout sizes
      const width = comp.properties.width || 250;
      const height = comp.properties.height || 60;

      const x = parentBounds.x + padding.left + safeArea.left;
      const y = currentY;

      // Assign bounds directly to component style properties
      comp.style.position = 'absolute';
      comp.style.left = `${x}px`;
      comp.style.top = `${y}px`;
      comp.style.width = `${width}px`;
      comp.style.height = `${height}px`;

      currentY += height + spacing;
    });

    return components;
  }
}

module.exports = ColumnLayoutStrategy;
