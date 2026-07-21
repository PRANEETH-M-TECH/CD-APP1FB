const Scene = require('../Scene');
const SceneNode = require('../SceneNode');

// Subnode mappings
const TextNode = require('../nodes/TextNode');
const ImageNode = require('../nodes/ImageNode');
const SvgNode = require('../nodes/SvgNode');
const GroupNode = require('../nodes/GroupNode');
const ShapeNode = require('../nodes/ShapeNode');
const LabelNode = require('../nodes/LabelNode');
const PlaceholderNode = require('../nodes/PlaceholderNode');
const CustomNode = require('../nodes/CustomNode');

/**
 * SceneGraphAPI.js
 * Extensible utility API to create, modify, and serialize scenes and nodes programmatically.
 */
module.exports = {
  createScene: (sceneNo, templateId, teacherScript = '', metadata = {}, timeline = {}) => {
    return new Scene(sceneNo, templateId, teacherScript, metadata, [], timeline);
  },

  createNode: (id, type, properties = {}, children = [], metadata = {}) => {
    switch (type.toUpperCase()) {
      case 'TEXT':
        return new TextNode(id, properties.text || '', properties, children, metadata);
      case 'IMAGE':
        return new ImageNode(id, properties.url || '', properties, children, metadata);
      case 'SVG':
        return new SvgNode(id, properties, children, metadata);
      case 'GROUP':
        return new GroupNode(id, properties, children, metadata);
      case 'SHAPE':
        return new ShapeNode(id, properties.shapeType || 'rect', properties, children, metadata);
      case 'LABEL':
        return new LabelNode(id, properties.text || '', properties.targetId || '', properties, children, metadata);
      case 'PLACEHOLDER':
        return new PlaceholderNode(id, properties, children, metadata);
      case 'CUSTOM':
      default:
        return new CustomNode(id, properties, children, metadata);
    }
  },

  addNode: (scene, node) => {
    scene.addNode(node);
  },

  removeNode: (scene, nodeId) => {
    return scene.removeNode(nodeId);
  },

  findNode: (scene, nodeId) => {
    return scene.findNode(nodeId);
  },

  traverse: (scene, callback) => {
    scene.traverse(callback);
  },

  serialize: (sceneOrGraph) => {
    return sceneOrGraph.serialize();
  },

  deserializeScene: (json) => {
    return Scene.deserialize(json);
  },

  deserializeNode: (json) => {
    return SceneNode.deserialize(json);
  }
};
