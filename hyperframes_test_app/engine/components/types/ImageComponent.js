const Component = require('../base/Component');

/**
 * ImageComponent.js
 * A reusable component representing image assets.
 */
class ImageComponent extends Component {
  constructor(id, properties = {}, style = {}, children = [], metadata = {}, visibility = true) {
    super(id, 'IMAGE', properties, style, children, metadata, visibility);
  }
}

module.exports = ImageComponent;
