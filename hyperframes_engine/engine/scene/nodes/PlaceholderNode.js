const SceneNode = require('../SceneNode');

/**
 * PlaceholderNode.js
 * A Scene Graph Node representing standard dynamic mock layout holders.
 */
class PlaceholderNode extends SceneNode {
  constructor(id, properties = {}, children = [], metadata = {}) {
    super(id, 'PLACEHOLDER', properties, children, metadata);
  }
}

module.exports = PlaceholderNode;
