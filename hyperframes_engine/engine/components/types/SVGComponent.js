const Component = require('../base/Component');

/**
 * SVGComponent.js
 * A reusable component representing direct SVG graphics or custom vector drawings.
 */
class SVGComponent extends Component {
  constructor(id, properties = {}, style = {}, children = [], metadata = {}, visibility = true) {
    super(id, 'SVG', properties, style, children, metadata, visibility);
  }
}

module.exports = SVGComponent;
