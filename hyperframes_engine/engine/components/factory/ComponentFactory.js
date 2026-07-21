const ComponentRegistry = require('../registry/ComponentRegistry');

/**
 * ComponentFactory.js
 * Factory responsible for instantiating Components with defaults assignment
 * and required properties validation.
 */
class ComponentFactory {
  /**
   * Creates a Component instance of the specified type.
   * @param {string} type 
   * @param {string} id 
   * @param {object} properties 
   * @param {object} style 
   * @param {Array} children 
   * @param {object} metadata 
   * @param {boolean} visibility 
   * @returns {Component}
   */
  static create(type, id, properties = {}, style = {}, children = [], metadata = {}, visibility = true) {
    const upperType = type.toUpperCase();
    
    // Fetch default properties and styles for each component type
    const defaults = ComponentFactory.getDefaultsForType(upperType);
    const mergedProperties = { ...defaults.properties, ...properties };
    const mergedStyle = { ...defaults.style, ...style };

    // Validate required fields
    ComponentFactory.validateProperties(upperType, mergedProperties);

    return ComponentRegistry.instantiate(
      upperType,
      id,
      mergedProperties,
      mergedStyle,
      children,
      metadata,
      visibility
    );
  }

  /**
   * Gets default configurations for each component type.
   * @param {string} type 
   * @returns {object}
   */
  static getDefaultsForType(type) {
    switch (type) {
      case 'TEXT':
        return {
          properties: { text: '' },
          style: { fontFamily: 'Inter', fontSize: '16px', color: '#FFFFFF' }
        };
      case 'IMAGE':
        return {
          properties: { url: '' },
          style: { width: '100%', height: '100%' }
        };
      case 'SHAPE':
        return {
          properties: { shapeType: 'rect' },
          style: { fill: 'none', stroke: '#FFFFFF', strokeWidth: 2 }
        };
      case 'ARROW':
        return {
          properties: { fromX: 0, fromY: 0, toX: 0, toY: 0 },
          style: { stroke: '#FFFFFF', strokeWidth: 2 }
        };
      case 'GROUP':
      default:
        return {
          properties: {},
          style: {}
        };
    }
  }

  /**
   * Asserts required property definitions for key component types.
   * @param {string} type 
   * @param {object} properties 
   */
  static validateProperties(type, properties) {
    if (type === 'TEXT' && properties.text === undefined) {
      console.warn(`[Validation Warning] TEXT component is missing a 'text' property.`);
    }
    if (type === 'IMAGE' && properties.url === undefined) {
      console.warn(`[Validation Warning] IMAGE component is missing a 'url' property.`);
    }
  }
}

module.exports = ComponentFactory;
