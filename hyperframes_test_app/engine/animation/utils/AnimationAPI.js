const Animation = require('../models/Animation');
const AnimationTimeline = require('../timeline/AnimationTimeline');
const GSAPGenerator = require('../generator/GSAPGenerator');
const AnimationAdapter = require('../adapters/AnimationAdapter');

/**
 * AnimationAPI.js
 * Developer-facing public API to manipulate animations and compile timelines.
 */
module.exports = {
  /**
   * Instantiates a new Animation.
   * @param {object} fields 
   * @returns {Animation}
   */
  createAnimation: (fields) => {
    return new Animation(fields);
  },

  /**
   * Adds an animation to a timeline.
   * @param {AnimationTimeline} timeline 
   * @param {Animation} anim 
   */
  addAnimation: (timeline, anim) => {
    timeline.addAnimation(anim);
  },

  /**
   * Removes an animation from a timeline.
   * @param {AnimationTimeline} timeline 
   * @param {string} animId 
   */
  removeAnimation: (timeline, animId) => {
    timeline.removeAnimation(animId);
  },

  /**
   * Generates GSAP script strings from a timeline.
   * @param {AnimationTimeline} timeline 
   * @returns {string}
   */
  play: (timeline) => {
    return GSAPGenerator.generateGSAP(timeline);
  },

  /**
   * Serializes a timeline to JSON.
   * @param {AnimationTimeline} timeline 
   * @returns {Array<object>}
   */
  serialize: (timeline) => {
    return timeline.serialize();
  },

  /**
   * Deserializes a timeline from JSON.
   * @param {Array<object>} json 
   * @returns {AnimationTimeline}
   */
  deserialize: (json) => {
    return AnimationTimeline.deserialize(json);
  },

  /**
   * Adapts legacy script block.
   * @param {string} script 
   * @returns {AnimationTimeline}
   */
  adaptLegacy: (script) => {
    return AnimationAdapter.adaptLegacy(script);
  }
};
