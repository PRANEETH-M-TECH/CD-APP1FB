const Component = require('../base/Component');

/**
 * ShapeComponent.js
 * A reusable component representing geometric shapes.
 */
class ShapeComponent extends Component {
  constructor(id, properties = {}, style = {}, children = [], metadata = {}, visibility = true) {
    super(id, 'SHAPE', properties, style, children, metadata, visibility);
  }
}

module.exports = ShapeComponent;
