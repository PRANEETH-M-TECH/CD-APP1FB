const Component = require('../base/Component');

/**
 * CustomComponent.js
 * A reusable component representing custom or plugin-defined components.
 */
class CustomComponent extends Component {
  constructor(id, properties = {}, style = {}, children = [], metadata = {}, visibility = true) {
    super(id, 'CUSTOM', properties, style, children, metadata, visibility);
  }
}

module.exports = CustomComponent;
