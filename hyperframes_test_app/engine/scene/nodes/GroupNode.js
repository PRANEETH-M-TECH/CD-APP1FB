const SceneNode = require('../SceneNode');

/**
 * GroupNode.js
 * A Scene Graph Node representing a logical grouping container for children.
 */
class GroupNode extends SceneNode {
  constructor(id, properties = {}, children = [], metadata = {}) {
    super(id, 'GROUP', properties, children, metadata);
  }
}

module.exports = GroupNode;
