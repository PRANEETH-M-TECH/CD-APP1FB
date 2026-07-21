const SceneNode = require('./SceneNode');

/**
 * Scene.js
 * Represents a single scene container inside a HyperFrames lesson.
 * Modified in Milestone 2 - Iteration 5 to support Theme references.
 */
class Scene {
  constructor(sceneNo, templateId, teacherScript = '', metadata = {}, nodes = [], timeline = {}, camera = null, layout = null, focuses = [], themeId = 'default') {
    this.sceneNo = sceneNo;
    this.templateId = templateId;
    this.teacherScript = teacherScript;
    this.metadata = metadata;
    this.nodes = nodes; // Root level SceneNodes
    this.timeline = timeline; // Duration, stagger styles, transitions
    this.camera = camera; // Optional Scene Camera instance
    this.layout = layout; // Optional Scene Layout config
    this.focuses = focuses; // Array of Focus configurations
    this.themeId = themeId; // Theme ID reference string
  }

  /**
   * Adds a root node to this scene.
   * @param {SceneNode} node 
   */
  addNode(node) {
    this.nodes.push(node);
  }

  /**
   * Recursively removes a node by ID from this scene's graph.
   * @param {string} nodeId 
   * @returns {SceneNode|null}
   */
  removeNode(nodeId) {
    const idx = this.nodes.findIndex(n => n.id === nodeId);
    if (idx !== -1) {
      return this.nodes.splice(idx, 1)[0];
    }
    for (const node of this.nodes) {
      const removed = node.removeChild(nodeId);
      if (removed) return removed;
    }
    return null;
  }

  /**
   * Recursively finds a node by ID in this scene's graph.
   * @param {string} nodeId 
   * @returns {SceneNode|null}
   */
  findNode(nodeId) {
    for (const node of this.nodes) {
      const found = node.findChild(nodeId);
      if (found) return found;
    }
    return null;
  }

  /**
   * Traverses all nodes in this scene's graph.
   * @param {function} callback 
   */
  traverse(callback) {
    for (const node of this.nodes) {
      node.traverse(callback);
    }
  }

  /**
   * Serializes the Scene into a pure JSON object.
   * @returns {object}
   */
  serialize() {
    return {
      scene_no: this.sceneNo,
      template_id: this.templateId,
      teacher_script: this.teacherScript,
      metadata: this.metadata,
      nodes: this.nodes.map(n => n.serialize()),
      timeline: this.timeline,
      camera: this.camera ? this.camera.serialize() : null,
      layout: this.layout ? this.layout.serialize() : null,
      focuses: this.focuses.map(f => f.serialize()),
      theme_id: this.themeId,
      // Legacy fields required by run-storyboard HTML/audio pipeline
      template_data: this.template_data || null,
      audio_url: this.audio_url || (this.timeline && this.timeline.audio_url) || null,
      durationInFrames: this.durationInFrames != null
        ? this.durationInFrames
        : (this.timeline && this.timeline.durationInFrames) || null
    };
  }

  /**
   * Deserializes a Scene structure from JSON.
   * @param {object} json 
   * @returns {Scene}
   */
  static deserialize(json) {
    if (!json) return null;
    const Camera = require('../camera/models/Camera');
    const Layout = require('../layout/models/Layout');
    const Focus = require('../focus/models/Focus');
    const nodes = (json.nodes || []).map(n => SceneNode.deserialize(n));
    const camera = json.camera ? Camera.deserialize(json.camera) : null;
    const layout = json.layout ? Layout.deserialize(json.layout) : null;
    const focuses = (json.focuses || []).map(f => Focus.deserialize(f));
    const scene = new Scene(
      json.scene_no,
      json.template_id,
      json.teacher_script,
      json.metadata || {},
      nodes,
      json.timeline || {},
      camera,
      layout,
      focuses,
      json.theme_id || 'default'
    );
    scene.template_data = json.template_data || null;
    scene.audio_url = json.audio_url || (json.timeline && json.timeline.audio_url) || null;
    scene.durationInFrames = json.durationInFrames != null
      ? json.durationInFrames
      : (json.timeline && json.timeline.durationInFrames) || null;
    return scene;
  }
}

module.exports = Scene;
