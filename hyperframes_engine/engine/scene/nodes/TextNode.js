const SceneNode = require('../SceneNode');

/**
 * TextNode.js
 * A Scene Graph Node representing text content (e.g. titles, labels, equations, etc.)
 */
class TextNode extends SceneNode {
  constructor(id, text = '', properties = {}, children = [], metadata = {}) {
    super(id, 'TEXT', { text, ...properties }, children, metadata);
  }
}

module.exports = TextNode;
