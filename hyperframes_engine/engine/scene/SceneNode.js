/**
 * SceneNode.js
 * Base class for all nodes in the HyperFrames Scene Graph.
 * Modified in Iteration 2 to contain reusable Component objects.
 */
class SceneNode {
  constructor(id, type, component = null, children = [], metadata = {}) {
    this.id = id || `node_${Math.random().toString(36).substr(2, 9)}`;
    this.type = type || 'CUSTOM';
    this.component = component; // The Component instance
    this.children = children;
    this.metadata = metadata;
  }

  /**
   * Adds a child node to this node.
   * @param {SceneNode} node 
   */
  addChild(node) {
    this.children.push(node);
  }

  /**
   * Recursively removes a child node by ID.
   * @param {string} nodeId 
   * @returns {SceneNode|null} The removed node, or null if not found.
   */
  removeChild(nodeId) {
    const idx = this.children.findIndex(c => c.id === nodeId);
    if (idx !== -1) {
      return this.children.splice(idx, 1)[0];
    }
    for (const child of this.children) {
      const removed = child.removeChild(nodeId);
      if (removed) return removed;
    }
    return null;
  }

  /**
   * Recursively finds a child node by ID.
   * @param {string} nodeId 
   * @returns {SceneNode|null}
   */
  findChild(nodeId) {
    if (this.id === nodeId) return this;
    for (const child of this.children) {
      const found = child.findChild(nodeId);
      if (found) return found;
    }
    return null;
  }

  /**
   * Traverses the node hierarchy depth-first.
   * @param {function} callback 
   */
  traverse(callback) {
    callback(this);
    for (const child of this.children) {
      child.traverse(callback);
    }
  }

  /**
   * Serializes the node structure to a pure JSON object.
   * @returns {object}
   */
  serialize() {
    return {
      id: this.id,
      type: this.type,
      component: this.component ? this.component.serialize() : null,
      children: this.children.map(c => c.serialize()),
      metadata: this.metadata
    };
  }

  /**
   * Deserializes a JSON object into a SceneNode structure.
   * @param {object} json 
   * @returns {SceneNode}
   */
  static deserialize(json) {
    if (!json) return null;
    
    const Component = require('../components/base/Component');
    const component = json.component ? Component.deserialize(json.component) : null;
    const children = (json.children || []).map(c => SceneNode.deserialize(c));
    
    return new SceneNode(
      json.id,
      json.type,
      component,
      children,
      json.metadata
    );
  }
}

module.exports = SceneNode;
