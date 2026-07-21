const Component = require('../base/Component');

/**
 * LabelComponent.js
 * A reusable component representing annotations or label markers.
 */
class LabelComponent extends Component {
  constructor(id, properties = {}, style = {}, children = [], metadata = {}, visibility = true) {
    super(id, 'LABEL', properties, style, children, metadata, visibility);
  }
}

module.exports = LabelComponent;
