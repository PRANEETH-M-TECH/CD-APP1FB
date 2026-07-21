const SceneNode = require('../SceneNode');

/**
 * LabelNode.js
 * A Scene Graph Node representing descriptive markers or labels referencing target nodes.
 */
class LabelNode extends SceneNode {
  constructor(id, text = '', targetId = '', properties = {}, children = [], metadata = {}) {
    super(id, 'LABEL', { text, targetId, ...properties }, children, metadata);
  }
}

module.exports = LabelNode;
