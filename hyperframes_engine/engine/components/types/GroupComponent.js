const Component = require('../base/Component');

/**
 * GroupComponent.js
 * A reusable component representing a nesting container for other components.
 */
class GroupComponent extends Component {
  constructor(id, properties = {}, style = {}, children = [], metadata = {}, visibility = true) {
    super(id, 'GROUP', properties, style, children, metadata, visibility);
  }
}

module.exports = GroupComponent;
