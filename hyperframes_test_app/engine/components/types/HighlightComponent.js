const Component = require('../base/Component');

/**
 * HighlightComponent.js
 * A reusable component representing highlights or cards focal effects.
 */
class HighlightComponent extends Component {
  constructor(id, properties = {}, style = {}, children = [], metadata = {}, visibility = true) {
    super(id, 'HIGHLIGHT', properties, style, children, metadata, visibility);
  }
}

module.exports = HighlightComponent;
