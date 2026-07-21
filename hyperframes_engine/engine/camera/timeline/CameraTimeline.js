/**
 * CameraTimeline.js
 * Stores keyframe definitions and transitions describing camera viewing behavior changes over time.
 * Does not execute visual animations directly.
 */
class CameraTimeline {
  constructor(keyframes = []) {
    this.keyframes = keyframes.map(kf => ({
      time: kf.time || 0, // Time offset (seconds or frames)
      position: kf.position || null, // Target position coordinate {x, y}
      zoom: kf.zoom || null, // Target zoom factor
      target: kf.target || null, // Node focus target
      transition: kf.transition || 'linear', // Transition type
      duration: kf.duration || 0, // Easing animation duration
      easing: kf.easing || 'none' // Easing styles (e.g. power2.inOut)
    }));
  }

  /**
   * Serializes the timeline configurations.
   * @returns {Array<object>}
   */
  serialize() {
    return this.keyframes;
  }
}

module.exports = CameraTimeline;
