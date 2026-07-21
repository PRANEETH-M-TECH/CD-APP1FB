const SceneNode = require('../SceneNode');

/**
 * ShapeNode.js
 * A Scene Graph Node representing standard geometric shapes (circles, rectangles, paths, lines).
 */
class ShapeNode extends SceneNode {
  constructor(id, shapeType = 'rect', properties = {}, children = [], metadata = {}) {
    super(id, 'SHAPE', { shapeType, ...properties }, children, metadata);
  }
}

module.exports = ShapeNode;
