const Component = require('../base/Component');

/**
 * TextComponent.js
 * A reusable component representing text blocks.
 */
class TextComponent extends Component {
  constructor(id, properties = {}, style = {}, children = [], metadata = {}, visibility = true) {
    super(id, 'TEXT', properties, style, children, metadata, visibility);
  }
}

module.exports = TextComponent;
