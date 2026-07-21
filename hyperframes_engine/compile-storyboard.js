const fs = require('fs');
const path = require('path');

// Import Hyperframes Core & Performance Subsystems
const HyperframesConfig = require('./engine/core/config/HyperframesConfig');
const Logger = require('./engine/core/logging/Logger');
const MetricsCollector = require('./engine/core/metrics/MetricsCollector');
const Diagnostics = require('./engine/core/diagnostics/Diagnostics');
const ValidationFramework = require('./engine/core/validation/ValidationFramework');
const PerformanceAPI = require('./engine/performance/interfaces/PerformanceAPI');
const StoryboardAdapter = require('./engine/adapters/StoryboardAdapter');
const ThemeManager = require('./engine/theme/manager/ThemeManager');

// Template Registries
const templates = {
  title_slide: require('./templates/TitleSlide'),
  concept_diagram: require('./templates/ConceptDiagram'),
  cycle_template: require('./templates/CycleTemplate'),
  math_derivation: require('./templates/MathDerivation'),
  column_comparison: require('./templates/ColumnComparison'),
  horizontal_timeline: require('./templates/HorizontalTimeline'),
  database_grid: require('./templates/DatabaseGrid'),
  venn_diagram: require('./templates/VennDiagram'),
  taxonomy_tree: require('./templates/TaxonomyTree'),
  cartesian_grid: require('./templates/CartesianGrid'),
  geo_marker: require('./templates/GeoMarker'),
  before_after_slider: require('./templates/BeforeAfterSlider'),
  quiz_checkpoint: require('./templates/QuizCheckpoint'),
  illustrated_scene: require('./templates/IllustratedScene'),
  image_scene: require('./templates/ImageScene'),
  general_scene: require('./templates/GeneralScene'),
};

// Initialize Core & Performance Configuration
const config = HyperframesConfig.load();
Logger.setLevel(config.logLevel);
PerformanceAPI.initialize(config);

function compileStoryboard(filePath) {
  let resolvedPath = path.resolve(filePath.trim());
  if (!fs.existsSync(resolvedPath)) {
    console.error(`[COMPILER ERROR] File not found: ${resolvedPath}`);
    process.exit(1);
  }

  let storyboardData;
  try {
    const rawJson = JSON.parse(fs.readFileSync(resolvedPath, 'utf8'));
    const sceneGraph = StoryboardAdapter.toSceneGraph(rawJson);
    storyboardData = sceneGraph.serialize();
  } catch (e) {
    console.error("[COMPILER ERROR] Could not parse or adapt JSON file:", e.message);
    process.exit(1);
  }

  const lessonDir = path.dirname(resolvedPath);

  // Import full HTML generator from run-storyboard helper
  const runStoryboardModule = require('./run-storyboard.js');
}

// CLI Entrypoint
const targetJson = process.argv[2];
if (!targetJson) {
  console.log("Usage: node compile-storyboard.js <path-to-storyboard.json>");
  process.exit(1);
}

// Call compile via processStoryboardPath if run-storyboard is executed with argument
