/**
 * CollisionDetector.js
 * Scans component bounds for overlapping coordinate bounding boxes
 * and yields displacement suggestions to resolve collisions.
 */
class CollisionDetector {
  /**
   * Checks if two coordinate bounding boxes overlap.
   * @param {object} boxA {x, y, width, height}
   * @param {object} boxB {x, y, width, height}
   * @returns {boolean}
   */
  static isOverlapping(boxA, boxB) {
    return (
      boxA.x < boxB.x + boxB.width &&
      boxA.x + boxA.width > boxB.x &&
      boxA.y < boxB.y + boxB.height &&
      boxA.y + boxA.height > boxB.y
    );
  }

  /**
   * Scans components inside a scene and returns vertical offset resolution suggestions.
   * @param {Array<Component>} components 
   * @returns {Array<object>} Suggestions array
   */
  static detectAndSuggest(components) {
    const suggestions = [];
    const bounds = (components || []).map((c) => {
      const x = parseFloat(c.style.left || c.properties.x || 0);
      const y = parseFloat(c.style.top || c.properties.y || 0);
      const w = parseFloat(c.style.width || c.properties.width || 100);
      const h = parseFloat(c.style.height || c.properties.height || 50);
      return { id: c.id, x, y, width: w, height: h };
    });

    for (let i = 0; i < bounds.length; i++) {
      for (let j = i + 1; j < bounds.length; j++) {
        if (CollisionDetector.isOverlapping(bounds[i], bounds[j])) {
          // Suggest displacing the second component vertically below the first one
          const shiftY = (bounds[i].y + bounds[i].height) - bounds[j].y + 10;
          suggestions.push({
            overlapping: [bounds[i].id, bounds[j].id],
            suggestion: {
              targetId: bounds[j].id,
              shiftX: 0,
              shiftY: shiftY
            }
          });
        }
      }
    }

    return suggestions;
  }
}

module.exports = CollisionDetector;
