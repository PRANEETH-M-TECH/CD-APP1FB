const RenderTreeNode = require('./RenderTreeNode');

/**
 * RenderTree.js
 * Internally compiles and structures RenderTreeNodes from the Scene Graph representation.
 */
class RenderTree {
  /**
   * Compiles the Scene Graph of a Scene into a modular Render Tree.
   * @param {Scene} scene 
   * @returns {RenderTree}
   */
  static build(scene) {
    const rootNodes = [];
    
    (scene.nodes || []).forEach((node) => {
      const isSvgContext = node.type === 'SVG';
      const treeNode = RenderTree.mapNode(node, isSvgContext);
      if (treeNode) {
        rootNodes.push(treeNode);
      }
    });

    const rt = new RenderTree();
    rt.rootNodes = rootNodes;
    return rt;
  }

  /**
   * Recursively maps a SceneNode to a RenderTreeNode.
   */
  static mapNode(node, svgContext) {
    if (!node || !node.component) return null;

    const rtNode = new RenderTreeNode(node, node.component, svgContext);
    
    // Map children recursively
    (node.children || []).forEach((childNode) => {
      // Propagate SVG context down the node tree
      const childSvgContext = svgContext || childNode.type === 'SVG';
      const childRtNode = RenderTree.mapNode(childNode, childSvgContext);
      if (childRtNode) {
        rtNode.addChild(childRtNode);
      }
    });

    return rtNode;
  }

  /**
   * Renders the entire Render Tree to HTML.
   * @returns {string}
   */
  render() {
    return this.rootNodes.map(r => r.render()).join('');
  }
}

module.exports = RenderTree;
