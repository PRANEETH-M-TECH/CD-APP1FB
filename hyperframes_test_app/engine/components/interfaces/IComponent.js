/**
 * IComponent.js
 * Interface declaration and validation helper for Component structures.
 * Guarantees that any class behaving as a Component implements required interface fields.
 */
class IComponent {
  /**
   * Asserts that a given instance implements the standard IComponent fields.
   * @param {object} instance 
   * @returns {boolean}
   * @throws {TypeError}
   */
  static validate(instance) {
    if (!instance) {
      throw new TypeError("Component validation failed: instance is null or undefined.");
    }
    
    const requiredMembers = [
      'id',
      'type',
      'properties',
      'style',
      'children',
      'metadata',
      'visibility'
    ];

    for (const member of requiredMembers) {
      if (instance[member] === undefined) {
        throw new TypeError(`Component validation failed: Required field '${member}' is missing.`);
      }
    }

    return true;
  }
}

module.exports = IComponent;
