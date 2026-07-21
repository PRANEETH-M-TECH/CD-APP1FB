/**
 * TransitionEngine.js
 * Central Transition Compiler mapping transition names into raw GSAP
 * entry animations sequentially stitched inside the main timeline.
 */
class TransitionEngine {
  /**
   * Generates GSAP transition code strings.
   * @param {string} type 
   * @param {string} targetSelector 
   * @param {number} duration 
   * @returns {string}
   */
  static generateTransition(type, targetSelector, duration = 0.5) {
    const d = duration;
    switch (type.toUpperCase()) {
      case 'FADE':
        return `      sceneTl.fromTo('${targetSelector}', { opacity: 0 }, { opacity: 1, duration: ${d}, ease: 'power2.out' });\n`;

      case 'SLIDE':
        return `      sceneTl.fromTo('${targetSelector}', { x: 1280 }, { x: 0, duration: ${d}, ease: 'power3.out' });\n`;

      case 'ZOOM':
        return `      sceneTl.fromTo('${targetSelector}', { scale: 0.8, opacity: 0 }, { scale: 1, opacity: 1, duration: ${d}, ease: 'back.out(1.5)' });\n`;

      case 'CUT':
      default:
        return `      sceneTl.set('${targetSelector}', { opacity: 1 });\n`;
    }
  }
}

module.exports = TransitionEngine;
