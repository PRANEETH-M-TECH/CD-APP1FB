const Component = require('../base/Component');

/**
 * ArrowComponent.js
 * A reusable component representing directional connections.
 */
class ArrowComponent extends Component {
  constructor(id, properties = {}, style = {}, children = [], metadata = {}, visibility = true) {
    super(id, 'ARROW', properties, style, children, metadata, visibility);
  }
}

module.exports = ArrowComponent;
