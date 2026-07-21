const SceneNode = require('../SceneNode');

/**
 * ImageNode.js
 * A Scene Graph Node representing image media assets.
 */
class ImageNode extends SceneNode {
  constructor(id, url = '', properties = {}, children = [], metadata = {}) {
    super(id, 'IMAGE', { url, ...properties }, children, metadata);
  }
}

module.exports = ImageNode;
