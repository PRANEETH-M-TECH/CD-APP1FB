const Theme = require('../models/Theme');

const registeredThemes = {};

/**
 * ThemeManager.js
 * Decoupled manager compiling design tokens into CSS Custom Properties
 * and registering available themes.
 */
class ThemeManager {
  /**
   * Registers a Theme configuration instance.
   * @param {string} id 
   * @param {Theme} theme 
   */
  static registerTheme(id, theme) {
    if (theme) {
      registeredThemes[id] = theme;
    }
  }

  /**
   * Retrieves a registered Theme configuration.
   * @param {string} id 
   * @returns {Theme}
   */
  static getTheme(id) {
    return registeredThemes[id] || registeredThemes['default'];
  }

  /**
   * Generates root-level CSS Variable mappings representing the active Theme tokens.
   * @param {Theme} theme 
   * @returns {string} Compiled CSS block
   */
  static getCSSVariables(theme) {
    const t = theme || ThemeManager.getTheme('default');
    const { colorPalette, typography, borderRadius, spacing, shadows } = t;

    return `
      :root {
        --theme-primary-color: ${colorPalette.primaryColor || '#0f172a'};
        --theme-secondary-color: ${colorPalette.secondaryColor || '#1e293b'};
        --theme-accent-color: ${colorPalette.accentColor || '#3b82f6'};
        --theme-bg-color: ${colorPalette.backgroundColor || '#090d16'};
        --theme-surface-color: ${colorPalette.surfaceColor || '#131b2e'};
        --theme-text-color: ${colorPalette.textColor || '#ffffff'};
        --theme-muted-text-color: ${colorPalette.mutedTextColor || 'rgba(255, 255, 255, 0.7)'};

        --theme-font-family: ${typography.fontFamily || 'Inter, system-ui, sans-serif'};
        --theme-font-size-title: ${typography.fontSizeTitle || '56px'};
        --theme-font-size-heading: ${typography.fontSizeHeading || '34px'};
        --theme-font-size-body: ${typography.fontSizeBody || '16px'};

        --theme-border-radius-sm: ${borderRadius.small || '4px'};
        --theme-border-radius-md: ${borderRadius.medium || '8px'};
        --theme-border-radius-lg: ${borderRadius.large || '16px'};

        --theme-spacing-sm: ${spacing.small || '8px'};
        --theme-spacing-md: ${spacing.medium || '16px'};
        --theme-spacing-lg: ${spacing.large || '24px'};

        --theme-shadow-low: ${shadows.low || '0 2px 4px rgba(0,0,0,0.1)'};
        --theme-shadow-high: ${shadows.high || '0 4px 16px rgba(0,0,0,0.4)'};
      }
    `;
  }
}

// Register default theme matching legacy styles
ThemeManager.registerTheme('default', new Theme());

module.exports = ThemeManager;
