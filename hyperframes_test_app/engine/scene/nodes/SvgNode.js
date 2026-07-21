const SceneNode = require('../SceneNode');

/**
 * SvgNode.js
 * A Scene Graph Node representing custom SVG vectors or canvases.
 */
class SvgNode extends SceneNode {
  constructor(id, properties = {}, children = [], metadata = {}) {
    super(id, 'SVG', properties, children, metadata);
  }
}

module.exports = SvgNode;
