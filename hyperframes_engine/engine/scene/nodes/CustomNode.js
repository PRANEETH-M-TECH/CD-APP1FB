const SceneNode = require('../SceneNode');

/**
 * CustomNode.js
 * A Scene Graph Node representing user-defined components or fallback custom types.
 */
class CustomNode extends SceneNode {
  constructor(id, properties = {}, children = [], metadata = {}) {
    super(id, 'CUSTOM', properties, children, metadata);
  }
}

module.exports = CustomNode;
