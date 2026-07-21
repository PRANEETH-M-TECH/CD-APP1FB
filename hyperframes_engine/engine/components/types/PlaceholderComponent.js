const Component = require('../base/Component');

/**
 * PlaceholderComponent.js
 * A reusable component representing dynamic mock visual positions.
 */
class PlaceholderComponent extends Component {
  constructor(id, properties = {}, style = {}, children = [], metadata = {}, visibility = true) {
    super(id, 'PLACEHOLDER', properties, style, children, metadata, visibility);
  }
}

module.exports = PlaceholderComponent;
