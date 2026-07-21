const registry = {};

/**
 * ComponentRegistry.js
 * Central decoupled registry managing mapping of component types to class definitions.
 * Eliminates switch statements and enables future plugin extensions.
 */
class ComponentRegistry {
  /**
   * Registers a Component type constructor.
   * @param {string} type 
   * @param {class} componentClass 
   */
  static register(type, componentClass) {
    registry[type.toUpperCase()] = componentClass;
  }

  /**
   * Retrieves a Component type constructor.
   * @param {string} type 
   * @returns {class|undefined}
   */
  static get(type) {
    return registry[type.toUpperCase()];
  }

  /**
   * Instantiates a Component instance dynamically.
   * @param {string} type 
   * @param {string} id 
   * @param {object} properties 
   * @param {object} style 
   * @param {Array} children 
   * @param {object} metadata 
   * @param {boolean} visibility 
   * @returns {Component}
   */
  static instantiate(type, id, properties = {}, style = {}, children = [], metadata = {}, visibility = true) {
    const ComponentClass = ComponentRegistry.get(type);
    if (!ComponentClass) {
      const CustomComponent = require('../types/CustomComponent');
      return new CustomComponent(id, properties, style, children, metadata, visibility);
    }
    return new ComponentClass(id, properties, style, children, metadata, visibility);
  }
}

module.exports = ComponentRegistry;

// Register core components dynamically on load to populate the registry
ComponentRegistry.register('TEXT', require('../types/TextComponent'));
ComponentRegistry.register('IMAGE', require('../types/ImageComponent'));
ComponentRegistry.register('SVG', require('../types/SVGComponent'));
ComponentRegistry.register('SHAPE', require('../types/ShapeComponent'));
ComponentRegistry.register('LABEL', require('../types/LabelComponent'));
ComponentRegistry.register('ARROW', require('../types/ArrowComponent'));
ComponentRegistry.register('GROUP', require('../types/GroupComponent'));
ComponentRegistry.register('HIGHLIGHT', require('../types/HighlightComponent'));
ComponentRegistry.register('PLACEHOLDER', require('../types/PlaceholderComponent'));
ComponentRegistry.register('CUSTOM', require('../types/CustomComponent'));
