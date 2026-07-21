/**
 * Theme.js
 * Decoupled, serializable Theme model representing visual design tokens.
 * Completely renderer-independent and transition-agnostic.
 */
class Theme {
  constructor(fields = {}) {
    this.themeId = fields.theme_id || `theme_${Math.random().toString(36).substr(2, 9)}`;
    this.name = fields.name || 'Default Dark Theme';

    // Color Palette Tokens
    this.colorPalette = fields.color_palette || {
      primaryColor: '#0f172a',
      secondaryColor: '#1e293b',
      accentColor: '#3b82f6',
      backgroundColor: '#090d16',
      surfaceColor: '#131b2e',
      textColor: '#ffffff',
      mutedTextColor: 'rgba(255, 255, 255, 0.7)'
    };

    // Typography Scale Tokens
    this.typography = fields.typography || {
      fontFamily: 'Inter, system-ui, sans-serif',
      fontSizeTitle: '56px',
      fontSizeHeading: '34px',
      fontSizeBody: '16px',
      fontWeightNormal: '400',
      fontWeightBold: '700',
      fontWeightHeavy: '900'
    };

    // Spacing Units
    this.spacing = fields.spacing || {
      unit: '8px',
      small: '8px',
      medium: '16px',
      large: '24px'
    };

    // Shadow Levels
    this.shadows = fields.shadows || {
      low: '0 2px 4px rgba(0,0,0,0.1)',
      high: '0 4px 16px rgba(0,0,0,0.4)'
    };

    // Border Radius Tokens
    this.borderRadius = fields.border_radius || {
      small: '4px',
      medium: '8px',
      large: '16px'
    };

    this.iconStyle = fields.icon_style || 'stroke';
    this.animationStyle = fields.animation_style || 'smooth';
    this.transitionStyle = fields.transition_style || 'FADE'; // CUT, FADE, SLIDE, WIPE, DISSOLVE, ZOOM
    this.metadata = fields.metadata || {};
  }

  /**
   * Serializes the Theme instance to a pure JSON object.
   * @returns {object}
   */
  serialize() {
    return {
      theme_id: this.themeId,
      name: this.name,
      color_palette: this.colorPalette,
      typography: this.typography,
      spacing: this.spacing,
      shadows: this.shadows,
      border_radius: this.borderRadius,
      icon_style: this.iconStyle,
      animation_style: this.animationStyle,
      transition_style: this.transitionStyle,
      metadata: this.metadata
    };
  }

  /**
   * Deserializes a Theme instance from a JSON object.
   * @param {object} json 
   * @returns {Theme}
   */
  static deserialize(json) {
    if (!json) return null;
    return new Theme(json);
  }
}

module.exports = Theme;
