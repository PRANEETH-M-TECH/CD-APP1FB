const Scene = require('./Scene');

/**
 * LessonSceneGraph.js
 * Root model representing the complete Scene Graph for a lesson storyboard.
 * Modified in Milestone 3 - Iteration 4 to support Pedagogical Strategy configuration.
 */
class LessonSceneGraph {
  constructor(title, theme, layoutMode, scenes = [], metadata = {}, teaching = null, narration = null, pedagogy = null) {
    this.title = title;
    this.theme = theme;
    this.layoutMode = layoutMode;
    this.scenes = scenes; // Array of Scene containers
    this.metadata = metadata;
    this.teaching = teaching; // Root TeachingModel plan
    this.narration = narration; // Root Narration timeline plan
    this.pedagogy = pedagogy; // Root PedagogicalStrategy configuration
  }

  /**
   * Serializes the LessonSceneGraph into a pure JSON storyboard object.
   * Keeps evolved fields while retaining full backward compatibility.
   * @returns {object}
   */
  serialize() {
    return {
      lesson_title: this.title,
      theme: this.theme,
      layout_mode: this.layoutMode,
      metadata: this.metadata,
      scenes: this.scenes.map(s => s.serialize()),
      teaching: this.teaching ? this.teaching.serialize() : null,
      narration: this.narration ? this.narration.serialize() : null,
      pedagogy: this.pedagogy ? this.pedagogy.serialize() : null
    };
  }

  /**
   * Deserializes a LessonSceneGraph from a JSON storyboard object.
   * @param {object} json 
   * @returns {LessonSceneGraph}
   */
  static deserialize(json) {
    if (!json) return null;
    const TeachingModel = require('../teaching/models/TeachingModel');
    const Narration = require('../synchronization/models/Narration');
    const PedagogicalStrategy = require('../pedagogy/models/PedagogicalStrategy');
    const scenes = (json.scenes || []).map(s => Scene.deserialize(s));
    const teaching = json.teaching ? TeachingModel.deserialize(json.teaching) : null;
    const narration = json.narration ? Narration.deserialize(json.narration) : null;
    const pedagogy = json.pedagogy ? PedagogicalStrategy.deserialize(json.pedagogy) : null;
    return new LessonSceneGraph(
      json.lesson_title || json.title,
      json.theme || 'indigo',
      json.layout_mode || 'process',
      scenes,
      json.metadata || {},
      teaching,
      narration,
      pedagogy
    );
  }
}

module.exports = LessonSceneGraph;
