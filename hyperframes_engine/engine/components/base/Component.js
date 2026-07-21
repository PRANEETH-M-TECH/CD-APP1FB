/**
 * Component.js
 * Base class for all reusable visual components in HyperFrames.
 * Components are lightweight, pure, renderer-agnostic, and layout-agnostic data models.
 */
class Component {
  constructor(id, type, properties = {}, style = {}, children = [], metadata = {}, visibility = true) {
    this.id = id || `comp_${Math.random().toString(36).substr(2, 9)}`;
    this.type = type || 'CUSTOM';
    this.properties = properties;
    this.style = style;
    this.children = children;
    this.metadata = metadata;
    this.visibility = visibility;
  }

  /**
   * Adds a child component to this component.
   * @param {Component} component 
   */
  addChild(component) {
    this.children.push(component);
  }

  /**
   * Serializes the component structure to a pure JSON object.
   * @returns {object}
   */
  serialize() {
    return {
      id: this.id,
      type: this.type,
      properties: this.properties,
      style: this.style,
      children: this.children.map(c => c.serialize()),
      metadata: this.metadata,
      visibility: this.visibility
    };
  }

  /**
   * Deserializes a JSON object into a Component structure using the Registry.
   * @param {object} json 
   * @returns {Component}
   */
  static deserialize(json) {
    if (!json) return null;
    
    // Resolve registry dynamically inside the function to prevent CommonJS circular imports
    const ComponentRegistry = require('../registry/ComponentRegistry');
    const children = (json.children || []).map(c => Component.deserialize(c));
    
    return ComponentRegistry.instantiate(
      json.type,
      json.id,
      json.properties,
      json.style,
      children,
      json.metadata,
      json.visibility
    );
  }
}

module.exports = Component;
