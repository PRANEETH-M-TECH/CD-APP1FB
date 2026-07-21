const Theme = require('../models/Theme');
const ThemeManager = require('../manager/ThemeManager');

/**
 * ThemeAPI.js
 * Developer-facing public API to manipulate themes and tokens.
 */
module.exports = {
  /**
   * Instantiates a new Theme.
   * @param {object} fields 
   * @returns {Theme}
   */
  createTheme: (fields) => {
    return new Theme(fields);
  },

  /**
   * Registers a Theme in ThemeManager.
   * @param {string} id 
   * @param {Theme} theme 
   */
  registerTheme: (id, theme) => {
    ThemeManager.registerTheme(id, theme);
  },

  /**
   * Retrieves a Theme.
   * @param {string} id 
   * @returns {Theme}
   */
  getTheme: (id) => {
    return ThemeManager.getTheme(id);
  },

  /**
   * Generates CSS Custom properties block.
   * @param {Theme} theme 
   * @returns {string}
   */
  getCSSVariables: (theme) => {
    return ThemeManager.getCSSVariables(theme);
  },

  /**
   * Serializes a Theme.
   * @param {Theme} theme 
   * @returns {object}
   */
  serialize: (theme) => {
    return theme.serialize();
  },

  /**
   * Deserializes a Theme from JSON.
   * @param {object} json 
   * @returns {Theme}
   */
  deserialize: (json) => {
    return Theme.deserialize(json);
  }
};
