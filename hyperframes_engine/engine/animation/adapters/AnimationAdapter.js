const Animation = require('../models/Animation');
const AnimationTimeline = require('../timeline/AnimationTimeline');

/**
 * AnimationAdapter.js
 * Backward compatibility adapter translating legacy string timelines into structured AnimationTimelines.
 */
class AnimationAdapter {
  /**
   * Adapts a legacy string GSAP animation block.
   * @param {string} legacyScript 
   * @returns {AnimationTimeline}
   */
  static adaptLegacy(legacyScript) {
    const timeline = new AnimationTimeline();
    if (typeof legacyScript === 'string' && legacyScript.trim()) {
      const anim = new Animation({
        type: 'CUSTOM',
        target: 'legacy',
        metadata: {
          script: legacyScript
        }
      });
      timeline.addAnimation(anim);
    }
    return timeline;
  }
}

module.exports = AnimationAdapter;
