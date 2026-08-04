// =============================================================
// HyperFrames Shared Icon Library
// Curated, self-contained set of 24x24 line-style SVG icons used to give
// every template (not just illustrated_scene) real pictorial content next
// to text, instead of shapes/boxes containing only labels. Each entry is
// the INNER markup of an <svg viewBox="0 0 24 24"> - no <svg> wrapper, no
// fill/stroke color baked in (callers apply the "theme-stroke" class so
// runtime theming colors it consistently with the rest of the engine).
//
// Keep icons stroke-based (fill="none", strokeWidth ~2, strokeLinecap/join
// "round") to match the existing icon-card / timeline-icon visual language
// already used by TitleSlide/HorizontalTimeline.
// =============================================================

var HFIcons = {
  // --- nature / science ---
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>',
  water_drop: '<path d="M12 2s7 8.5 7 13a7 7 0 0 1-14 0c0-4.5 7-13 7-13z"/>',
  cloud: '<path d="M6.5 19a4.5 4.5 0 0 1-.5-8.98A6 6 0 0 1 17.6 8.03 4.5 4.5 0 0 1 17 19H6.5z"/>',
  rain: '<path d="M6.5 15a4.5 4.5 0 0 1-.5-8.98A6 6 0 0 1 17.6 4.03 4.5 4.5 0 0 1 17 15H6.5z"/><path d="M8 18l-1 3M12 18l-1 3M16 18l-1 3"/>',
  wave: '<path d="M2 12c1.5-2 3.5-2 5 0s3.5 2 5 0 3.5-2 5 0 3.5 2 5 0"/><path d="M2 17c1.5-2 3.5-2 5 0s3.5 2 5 0 3.5-2 5 0 3.5 2 5 0"/>',
  ocean_wave: '<path d="M2 8c1.5-2 3.5-2 5 0s3.5 2 5 0 3.5-2 5 0 3.5 2 5 0"/><path d="M2 13c1.5-2 3.5-2 5 0s3.5 2 5 0 3.5-2 5 0 3.5 2 5 0"/><path d="M2 18c1.5-2 3.5-2 5 0s3.5 2 5 0 3.5-2 5 0 3.5 2 5 0"/>',
  water_tap: '<path d="M4 6h9a4 4 0 0 1 4 4v2"/><circle cx="4" cy="6" r="1.4"/><path d="M12 14v2.5"/><path d="M12 16.5a2.2 2.2 0 1 0 0 4.4 2.2 2.2 0 0 0 0-4.4z"/>',
  leaf: '<path d="M11 20A7 7 0 0 1 4 13c0-6 7-11 7-11s7 5 7 11a7 7 0 0 1-7 7z"/><path d="M11 20v-9"/>',
  tree: '<path d="M12 22v-6"/><path d="M12 16c-4 0-6-2.5-6-5.5C6 6.5 12 2 12 2s6 4.5 6 8.5c0 3-2 5.5-6 5.5z"/>',
  flower: '<circle cx="12" cy="12" r="2.5"/><path d="M12 2a3 3 0 0 1 3 3 3 3 0 0 1-3 3 3 3 0 0 1-3-3 3 3 0 0 1 3-3zM12 16a3 3 0 0 1 3 3 3 3 0 0 1-3 3 3 3 0 0 1-3-3 3 3 0 0 1 3-3zM4 12a3 3 0 0 1 3-3 3 3 0 0 1 3 3 3 3 0 0 1-3 3 3 3 0 0 1-3-3zM14 12a3 3 0 0 1 3-3 3 3 0 0 1 3 3 3 3 0 0 1-3 3 3 3 0 0 1-3-3z"/>',
  mountain: '<path d="M3 20l6-11 4 6 3-4 5 9H3z"/>',
  wind: '<path d="M3 8h11a3 3 0 1 0-3-3M3 12h15a3 3 0 1 1-3 3M3 16h9a2.5 2.5 0 1 1-2.5 2.5"/>',
  fire: '<path d="M12 22a6 6 0 0 0 6-6c0-3-2-5-3-7-.5 2-1.5 3-2.5 3C13 9 13 5 10.5 2 10 5 7 7 7 11a5 5 0 0 0 0 5 6 6 0 0 0 5 6z"/>',
  snowflake: '<path d="M12 2v20M4.2 7l15.6 10M4.2 17L19.8 7"/>',
  seed: '<path d="M12 2C7 2 4 7 4 12s3 10 8 10 8-5 8-10-3-10-8-10z"/><path d="M12 6v12"/>',
  atom: '<circle cx="12" cy="12" r="1.5"/><ellipse cx="12" cy="12" rx="10" ry="4.2"/><ellipse cx="12" cy="12" rx="10" ry="4.2" transform="rotate(60 12 12)"/><ellipse cx="12" cy="12" rx="10" ry="4.2" transform="rotate(120 12 12)"/>',
  molecule: '<circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="12" cy="17" r="2.5"/><path d="M8 7.5l3 8M16 7.5l-3 8M8.5 6h7"/>',

  // --- body / anatomy ---
  heart: '<path d="M12 21s-7.5-5-10-9.5C0.3 7.8 2.5 4 6.2 4c2 0 3.6 1.2 4.8 3 1.2-1.8 2.8-3 4.8-3 3.7 0 5.9 3.8 4.2 7.5C19.5 16 12 21 12 21z"/>',
  brain: '<path d="M9 3a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5h1a3 3 0 0 0 3-3V6a3 3 0 0 0-1-3z"/><path d="M15 3a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5h-1a3 3 0 0 1-3-3V6a3 3 0 0 1 1-3z"/>',
  lungs: '<path d="M12 3v9"/><path d="M12 12c-1-3-3-4-5-4-2.5 0-4 2-4 5 0 3 1.5 6 4 6 1.5 0 3-1 3.5-3"/><path d="M12 12c1-3 3-4 5-4 2.5 0 4 2 4 5 0 3-1.5 6-4 6-1.5 0-3-1-3.5-3"/>',
  eye: '<path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6z"/><circle cx="12" cy="12" r="3"/>',
  dna: '<path d="M6 3c0 6 12 12 12 18M18 3c0 6-12 12-12 18"/><path d="M7.5 7h9M6 12h12M7.5 17h9"/>',
  bone: '<path d="M5 9a2.5 2.5 0 1 1 3.5 2.3l7.2 7.2A2.5 2.5 0 1 1 18 21l-7.2-7.2A2.5 2.5 0 1 1 8.5 15l-2-2A2.5 2.5 0 0 1 5 9z"/>',

  // --- concepts / general ---
  lightbulb: '<path d="M9 18h6M10 21h4"/><path d="M12 3a6 6 0 0 0-4 10.5c.6.6 1 1.4 1 2.5h6c0-1.1.4-1.9 1-2.5A6 6 0 0 0 12 3z"/>',
  book: '<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v16H6.5A2.5 2.5 0 0 0 4 21.5v-16z"/><path d="M4 19a2.5 2.5 0 0 1 2.5-2.5H20"/>',
  pencil: '<path d="M17 3l4 4L7 21H3v-4L17 3z"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.6-1.1 1.7 1.7 0 0 0-.3-1.9l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.9.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.9-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.9V9c.3.6.9 1 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
  flag: '<path d="M5 3v18"/><path d="M5 4h13l-3 4 3 4H5"/>',
  trophy: '<path d="M8 4h8v5a4 4 0 0 1-8 0V4z"/><path d="M8 5H5a3 3 0 0 0 3 4M16 5h3a3 3 0 0 1-3 4"/><path d="M12 13v4M9 21h6M10 17h4v4h-4z"/>',
  star: '<path d="M12 2l3 7h7l-5.5 4.2L18.5 21 12 16.5 5.5 21l2-7.8L2 9h7z"/>',
  check: '<path d="M4 12l6 6L20 6"/>',
  cross: '<path d="M5 5l14 14M19 5L5 19"/>',
  arrow_up: '<path d="M12 20V4M5 11l7-7 7 7"/>',
  arrow_down: '<path d="M12 4v16M19 13l-7 7-7-7"/>',
  arrow_right: '<path d="M4 12h16M13 5l7 7-7 7"/>',
  arrow_left: '<path d="M20 12H4M11 19l-7-7 7-7"/>',
  cycle: '<path d="M4 12a8 8 0 0 1 14-5.3M20 12a8 8 0 0 1-14 5.3"/><path d="M18 3v4h-4M6 21v-4h4"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/>',
  calendar: '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/>',
  map_pin: '<path d="M12 22s7-6.5 7-12a7 7 0 1 0-14 0c0 5.5 7 12 7 12z"/><circle cx="12" cy="10" r="2.5"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
  home: '<path d="M4 11l8-7 8 7"/><path d="M6 10v10h12V10"/>',
  factory: '<path d="M3 21V11l5 3v-3l5 3V8l6 4v9H3z"/>',
  coin: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2 0 0 1 5 0c0 1.5-2.5 2-2.5 3.5M12 16v1"/>',
  scale: '<path d="M12 3v18M5 7l7-4 7 4"/><path d="M3 7h6M15 7h6"/><path d="M3 7l-2 6a4 4 0 0 0 8 0l-2-6zM21 7l-2 6a4 4 0 0 0 8 0l-2-6z"/>',
  people: '<circle cx="8" cy="8" r="3"/><path d="M2 20a6 6 0 0 1 12 0"/><circle cx="17" cy="9" r="2.5"/><path d="M14.5 20a5 5 0 0 1 7.5-4.3"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
  message: '<path d="M4 4h16v12H8l-4 4V4z"/>',
  shield: '<path d="M12 3l7 3v6c0 5-3.5 8-7 9-3.5-1-7-4-7-9V6l7-3z"/>',
  key: '<circle cx="8" cy="14" r="4"/><path d="M11 11l9-9M17 5l3 3M14 8l2 2"/>',
  lock: '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
  chart_bar: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
  chart_line: '<path d="M3 17l5-5 4 4 8-9"/><path d="M2 21h20"/>',
  database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  magnet: '<path d="M6 4h5v9a3.5 3.5 0 0 0 7 0V4h-5"/><path d="M6 4v9a3.5 3.5 0 0 1 0 0"/><path d="M6 8H2M22 8h-4"/>',
  battery: '<rect x="2" y="8" width="17" height="8" rx="1.5"/><path d="M21 11v2"/><path d="M6 11v2"/>',
  wire: '<path d="M3 12h4l2-4 4 8 2-4h6"/><circle cx="3" cy="12" r="1.3"/><circle cx="21" cy="12" r="1.3"/>',
  zap: '<path d="M13 2L4 14h6l-1 8 9-12h-6l1-8z"/>',
  ruler: '<rect x="3" y="9" width="18" height="6" rx="1"/><path d="M7 9v2.5M11 9v3M15 9v2.5M19 9v3"/>',
  filter: '<path d="M4 4h16l-6.5 8v6l-3 2v-8z"/>',
  rocket: '<path d="M12 2s5 3 5 9-5 11-5 11-5-8-5-11 5-9 5-9z"/><circle cx="12" cy="10" r="2"/><path d="M8 17l-3 3M16 17l3 3"/>',
  plane: '<path d="M3 12l18-7-7 18-2-8-8-2 -1-1z"/>',
  car: '<path d="M4 16V11l2-5h12l2 5v5"/><path d="M4 16h16"/><circle cx="7.5" cy="17.5" r="1.5"/><circle cx="16.5" cy="17.5" r="1.5"/>',
  boat: '<path d="M3 15h18l-2 5H5l-2-5z"/><path d="M6 15V6h6l4 9"/><path d="M6 6h0"/>',
  paw: '<circle cx="7" cy="7" r="1.7"/><circle cx="12" cy="5" r="1.7"/><circle cx="17" cy="7" r="1.7"/><path d="M12 12a5 5 0 0 1 5 5c0 2-2 3-5 3s-5-1-5-3a5 5 0 0 1 5-5z"/>',
  fish: '<path d="M3 12s4-5 11-5 7 5 7 5-1 5-7 5-11-5-11-5z"/><circle cx="17" cy="10.5" r="0.6"/><path d="M3 12l-2-3M3 12l-2 3"/>',
  bird: '<path d="M4 12c3-4 8-6 12-4 2 1 4 3 4 3l-4 1 1 3-5-1-3 3-1-3-4-2z"/>',
  bug: '<circle cx="12" cy="13" r="5"/><path d="M9 8V6M15 8V6M6 11l-2-1M18 11l2-1M6 15l-2 1M18 15l2 1M12 8v10"/>',
  hourglass: '<path d="M6 2h12M6 22h12"/><path d="M7 2c0 5 5 6 5 10s-5 5-5 10M17 2c0 5-5 6-5 10s5 5 5 10"/>',
  wave_hand: '<path d="M8 12V6a1.5 1.5 0 0 1 3 0v5M11 11V4a1.5 1.5 0 0 1 3 0v7M14 11V6a1.5 1.5 0 0 1 3 0v6M17 12v-3a1.5 1.5 0 0 1 3 0v6a6 6 0 0 1-6 6h-2a6 6 0 0 1-5-2.7L4 14"/>',
  handshake: '<path d="M2 12l4-4h4l2 2 2-2h4l4 4"/><path d="M8 12l3 3 2-2 2 2 3-3"/>',
  question: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 0 1 5 0c0 1.7-2.5 2-2.5 4"/><path d="M12 17v.01"/>',
  bell: '<path d="M6 10a6 6 0 0 1 12 0c0 5 2 6 2 6H4s2-1 2-6z"/><path d="M10 20a2 2 0 0 0 4 0"/>',
  compass: '<circle cx="12" cy="12" r="9"/><path d="M15 9l-2 6-6 2 2-6 6-2z"/>',
  dot: '<circle cx="12" cy="12" r="4"/>'
};

// Returns the inner SVG markup for a given icon name, falling back to a
// plain dot so a missing/unrecognized icon name never breaks rendering.
function getIconMarkup(name) {
  if (!name) return HFIcons.dot;
  var key = String(name).toLowerCase().trim().replace(/[\s-]+/g, '_');
  return HFIcons[key] || HFIcons.dot;
}

// Dual export: usable via require() at HTML-compile time in Node (Renderer.js
// embeds icon markup directly into the generated HTML string) and via
// <script src="./shared/icons.js"> in the browser, matching theme.js/animations.js.
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { HFIcons: HFIcons, getIconMarkup: getIconMarkup };
} else {
  window.HFIcons = HFIcons;
  window.getIconMarkup = getIconMarkup;
}
