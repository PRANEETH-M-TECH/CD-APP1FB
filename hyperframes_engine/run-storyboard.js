const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { spawn } = require('child_process');

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

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const outputsDir = path.join(__dirname, 'outputs');

// ============================================================================
// CORE LAYER: Production Configuration, Logging, Metrics, and Diagnostics
// ============================================================================
const HyperframesConfig = require('./engine/core/config/HyperframesConfig');
const Logger = require('./engine/core/logging/Logger');
const MetricsCollector = require('./engine/core/metrics/MetricsCollector');
const Diagnostics = require('./engine/core/diagnostics/Diagnostics');
const ValidationFramework = require('./engine/core/validation/ValidationFramework');

// Initialize configuration
const config = HyperframesConfig.load();
Logger.setLevel(config.logLevel);
Logger.info('Hyperframes engine starting', { env: config.env, logLevel: config.logLevel });

// Run startup diagnostics
if (config.features.diagnosticsOnStartup) {
  const diagReport = Diagnostics.report(config);
  if (diagReport.healthy) {
    Logger.info('Startup diagnostics passed', { summary: diagReport.summary });
  } else {
    Logger.warn('Startup diagnostics found issues', { summary: diagReport.summary, checks: diagReport.checks.filter(c => c.status !== 'ok') });
  }
}

// ============================================================================
// PERFORMANCE LAYER: Caching, Scheduling, Pooling, Profiling, Benchmarking
// ============================================================================
const PerformanceAPI = require('./engine/performance/interfaces/PerformanceAPI');
PerformanceAPI.initialize(config);
Logger.info('Performance layer initialized', { caches: PerformanceAPI.diagnostics().caches.map(c => c.name) });

function main() {
  const argPath = process.argv[2];
  const argAction = process.argv[3] || '2';
  if (argPath) {
    processStoryboardPath(argPath, argAction);
    return;
  }

  console.log("=========================================");
  console.log("   HYPERFRAMES STORYBOARD SELECTOR       ");
  console.log("=========================================");
  
  rl.question("Enter storyboard JSON file path (or press Enter to list generated storyboards): ", (pathInput) => {
    const filePath = pathInput.trim();
    if (filePath) {
      processStoryboardPath(filePath);
    } else {
      selectFromGenerated();
    }
  });
}

function processStoryboardPath(filePath, defaultAction = null) {
  let resolvedPath = filePath.trim();
  // Strip surrounding quotes
  if ((resolvedPath.startsWith('"') && resolvedPath.endsWith('"')) || (resolvedPath.startsWith("'") && resolvedPath.endsWith("'"))) {
    resolvedPath = resolvedPath.slice(1, -1);
  }

  if (!fs.existsSync(resolvedPath)) {
    // Check relative to project root
    const absPath = path.resolve(resolvedPath);
    if (!fs.existsSync(absPath)) {
      console.log(`[ERROR] File does not exist at: ${resolvedPath}`);
      rl.close();
      return;
    }
    resolvedPath = absPath;
  }

  let storyboardData;
  try {
    const rawJson = JSON.parse(fs.readFileSync(resolvedPath, 'utf8'));
    const StoryboardAdapter = require('./engine/adapters/StoryboardAdapter');
    const sceneGraph = StoryboardAdapter.toSceneGraph(rawJson);
    storyboardData = sceneGraph.serialize();
  } catch (e) {
    console.log("[ERROR] Could not parse or adapt JSON file:", e.message);
    rl.close();
    return;
  }

  const lessonId = storyboardData.lesson_id || path.basename(path.dirname(resolvedPath)) || 'custom_lesson';
  const lessonDir = path.dirname(resolvedPath);
  
  generateMasterHtml(storyboardData, lessonDir, (masterHtmlPath) => {
    if (defaultAction) {
      executeAction(defaultAction, masterHtmlPath, lessonId);
    } else {
      promptAction(masterHtmlPath, lessonId);
    }
  });
}

function executeAction(action, masterHtmlPath, lessonId) {
  console.log("\n======================================================================");
  console.log("🚀 [PIPELINE DEBUG] ENTER Renderer");
  console.log(`   Compiled master composition HTML at: ${masterHtmlPath}`);
  console.log("======================================================================\n");

  if (action === 'compile' || action === '0') {
    console.log("======================================================================");
    console.log("⚠️ [PIPELINE DEBUG] Video Generation: SKIPPED (Fast-path HTML compilation active)");
    console.log("======================================================================\n");
    rl.close();
    return;
  }

  console.log("======================================================================");
  console.log("🚀 [PIPELINE DEBUG] ENTER Video Generation");
  console.log("======================================================================\n");

  let cmd = 'npx';
  let args = [];
  const lessonDir = path.dirname(masterHtmlPath);
  const outName = path.join(lessonDir, `${lessonId}.mp4`);
  const customOutName = path.join(lessonDir, `custom_lesson.mp4`);

  if (action === '1') {
    console.log(`\nLaunching HyperFrames Preview on localhost...`);
    args = ['hyperframes', 'preview', lessonDir];
  } else {
    console.log(`\nRendering video to ${outName} (this might take a minute)...`);
    args = ['hyperframes', 'render', lessonDir, '--output', outName];
  }

  rl.close();

  const child = spawn(cmd, args, { 
    shell: true, 
    stdio: 'inherit',
    cwd: __dirname
  });

  child.on('close', (code) => {
    console.log(`\nProcess finished with exit code ${code}`);
    if (code === 0 && action === '2' && fs.existsSync(outName)) {
      try {
        fs.copyFileSync(outName, customOutName);
        console.log(`Copied rendered video to ${customOutName}`);
      } catch (e) {}
    }
  });
}

function selectFromGenerated() {
  if (!fs.existsSync(outputsDir)) {
    fs.mkdirSync(outputsDir, { recursive: true });
  }

  const dirs = fs.readdirSync(outputsDir).filter(f => {
    return fs.statSync(path.join(outputsDir, f)).isDirectory();
  });

  if (dirs.length === 0) {
    console.log("\nNo generated storyboards found in:");
    console.log(outputsDir);
    console.log("\nGenerate a visual lesson storyboard first!");
    rl.close();
    return;
  }

  const lessons = [];
  dirs.forEach((dir) => {
    const jsonPath = path.join(outputsDir, dir, 'lesson.json');
    if (fs.existsSync(jsonPath)) {
      try {
        const data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
        lessons.push({
          id: dir,
          title: data.lesson_title || "Unnamed Lesson",
          theme: data.theme || "indigo",
          scenesCount: data.scenes ? data.scenes.length : 0,
          filePath: jsonPath,
          lessonDir: path.join(outputsDir, dir)
        });
      } catch (err) {
        lessons.push({ id: dir, title: "Invalid lesson.json data", scenesCount: 0 });
      }
    } else {
      lessons.push({ id: dir, title: "Missing lesson.json", scenesCount: 0 });
    }
  });

  console.log("\nAvailable Storyboards:");
  lessons.forEach((l, idx) => {
    console.log(`[${idx + 1}] ID: ${l.id}`);
    console.log(`    Title: ${l.title}`);
    console.log(`    Theme: ${l.theme} | Scenes: ${l.scenesCount}`);
    console.log("-----------------------------------------");
  });

  rl.question(`Select a storyboard (1-${lessons.length}): `, (answer) => {
    const selection = parseInt(answer.trim(), 10);
    if (isNaN(selection) || selection < 1 || selection > lessons.length) {
      console.log("Invalid selection. Exiting.");
      rl.close();
      return;
    }

    const selectedLesson = lessons[selection - 1];
    if (!selectedLesson.filePath) {
      console.log("[ERROR] Selected lesson is invalid or missing file.");
      rl.close();
      return;
    }

    console.log(`\nYou selected: ${selectedLesson.title} (${selectedLesson.id})`);
    
    // Parse the storyboard data
    const rawJson = JSON.parse(fs.readFileSync(selectedLesson.filePath, 'utf8'));
    const StoryboardAdapter = require('./engine/adapters/StoryboardAdapter');
    const sceneGraph = StoryboardAdapter.toSceneGraph(rawJson);
    const storyboardData = sceneGraph.serialize();
    
    generateMasterHtml(storyboardData, selectedLesson.lessonDir, (masterHtmlPath) => {
      promptAction(masterHtmlPath, selectedLesson.id);
    });
  });
}

function promptAction(masterHtmlPath, lessonId) {
  console.log("\nChoose Action:");
  console.log("[1] Preview on Localhost (Browser Player)");
  console.log("[2] Render/Export to MP4 Video File");

  rl.question(`Select action (1-2): `, (actionAnswer) => {
    const action = actionAnswer.trim();
    let cmd = 'npx';
    let args = [];

    if (action === '1') {
      const lessonDir = path.dirname(masterHtmlPath);
      console.log(`\nLaunching HyperFrames Preview on localhost...`);
      args = ['hyperframes', 'preview', lessonDir];
    } else if (action === '2') {
      const lessonDir = path.dirname(masterHtmlPath);
      const outName = path.join(lessonDir, `${lessonId}.mp4`);
      console.log(`\nRendering video to ${outName} (this might take a minute)...`);
      args = ['hyperframes', 'render', lessonDir, '--output', outName];
    } else {
      console.log("Invalid action. Exiting.");
      rl.close();
      return;
    }

    rl.close();

    const child = spawn(cmd, args, { 
      shell: true, 
      stdio: 'inherit',
      cwd: __dirname
    });

    child.on('close', (code) => {
      console.log(`\nProcess finished with exit code ${code}`);
    });
  });
}

// ============================================================================
// HTML GENERATOR: Compiles the 16 templates into a single mega-HTML composition
// ============================================================================

function generateMasterHtml(storyboard, lessonDir, callback) {
  PerformanceAPI.getProfiler().start('generate_html');
  MetricsCollector.start('generate_html');
  Logger.info('HTML generation pipeline started', { lessonId: storyboard.lesson_id || 'unknown', sceneCount: (storyboard.scenes || []).length });

  // Run storyboard validation
  if (config.features.storyboardValidation) {
    const validationResult = ValidationFramework.validateStoryboard(storyboard);
    if (!validationResult.isValid) {
      Logger.warn('Storyboard validation found issues', { errors: validationResult.errors });
    } else {
      Logger.debug('Storyboard validation passed');
    }
    MetricsCollector.increment('validation_runs');
    if (!validationResult.isValid) MetricsCollector.increment('validation_warnings', validationResult.errors.length);
  }

  const ThemeManager = require('./engine/theme/manager/ThemeManager');
  const themeCSS = ThemeManager.getCSSVariables(null);

  const fps = 30;
  const scenes = storyboard.scenes || [];

  // Helper to read exact WAV audio file duration in seconds from header
  const getAudioDurationInSeconds = (scene) => {
    let audioUrl = scene.audio_url || (scene.timeline && scene.timeline.audio_url) || null;
    const fallbackName = `scene_${scene.scene_no}.wav`;
    let targetPath = path.join(lessonDir, fallbackName);
    
    if (audioUrl) {
      const fileName = path.basename(audioUrl);
      const possiblePath = path.join(lessonDir, fileName);
      if (fs.existsSync(possiblePath)) {
        targetPath = possiblePath;
      }
    }

    if (fs.existsSync(targetPath)) {
      try {
        const buf = fs.readFileSync(targetPath);
        if (buf.length > 44) {
          const byteRate = buf.readUInt32LE(28);
          if (byteRate > 0) {
            const wavSecs = (buf.length - 44) / byteRate;
            if (wavSecs > 0.5) return wavSecs;
          }
        }
      } catch (e) {
        // Fallback to estimation if file read fails
      }
    }
    return null;
  };

  // Estimate scene durations accurately based on exact audio duration + 0.3s transition padding
  let currentStart = 0;
  const scenesWithTiming = scenes.map((scene, idx) => {
    let duration = scene.durationInFrames ? scene.durationInFrames / fps : null;
    
    if (!duration) {
      const audioDuration = getAudioDurationInSeconds(scene);
      if (audioDuration) {
        duration = audioDuration + 0.3;
      } else {
        const words = scene.teacher_script ? scene.teacher_script.split(/\s+/).length : 0;
        duration = Math.max(3.0, words * 0.35 + 0.5);
      }
    }

    const sceneStart = currentStart;
    currentStart += duration;
    return {
      ...scene,
      start: sceneStart,
      duration: duration
    };
  });

  const totalDuration = currentStart;

  // Let's copy theme.js and animations.js locally to the lesson folder so it resolves cleanly
  const sharedDir = path.join(__dirname, 'shared');
  const targetSharedDir = path.join(lessonDir, 'shared');
  if (!fs.existsSync(targetSharedDir)) {
    fs.mkdirSync(targetSharedDir, { recursive: true });
  }
  fs.copyFileSync(path.join(sharedDir, 'theme.js'), path.join(targetSharedDir, 'theme.js'));
  fs.copyFileSync(path.join(sharedDir, 'animations.js'), path.join(targetSharedDir, 'animations.js'));

  // Build the master HTML content
  let html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>${storyboard.lesson_title || 'Visual Storyboard Video'}</title>
  
  <!-- CSS Fonts -->
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;700;900&family=Space+Grotesk:wght@400;700&family=Inter:wght@400;500;700;900&family=Cinzel:wght@700&family=Playfair+Display:wght@700&family=Roboto:wght@400;700&display=swap" rel="stylesheet">
  
  <!-- KaTeX for math rendering -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  
  <!-- GSAP for animations -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
  
  <!-- Shared hyperframes JS -->
  <script src="./shared/theme.js"></script>
  <script src="./shared/animations.js"></script>

  <style>
    ${themeCSS}
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body {
      width: 1280px;
      height: 720px;
      overflow: hidden;
      background: var(--theme-bg-color, #090d16);
      font-family: var(--theme-font-family, 'Inter', system-ui, sans-serif);
      color: var(--theme-text-color, #ffffff);
      -webkit-font-smoothing: antialiased;
    }
    
    .composition {
      width: 1280px;
      height: 720px;
      position: relative;
    }
    
    /* Scene wrappers managed by HyperFrames */
    .scene {
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
      z-index: 1;
    }

    /* Subtitles banner */
    .subtitles-container {
      display: none !important;
      position: absolute;
      bottom: 45px;
      left: 8%;
      right: 8%;
      text-align: center;
      font-size: 22px;
      font-weight: 700;
      line-height: 1.4;
      color: #ffffff;
      z-index: 90;
      text-shadow: 
        0 2px 4px rgba(0,0,0,0.9), 
        0 0 10px rgba(0,0,0,0.9), 
        1px 1px 0px #000, 
        -1px -1px 0px #000, 
        1px -1px 0px #000, 
        -1px 1px 0px #000;
      font-family: 'Inter', system-ui, sans-serif;
    }

    /* Template components specific styles */
    .title-slide-container {
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 60px;
    }
    .icon-card {
      width: 140px;
      height: 140px;
      border-radius: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 30px;
      margin-bottom: 40px;
    }
    .icon-card svg {
      width: 100%;
      height: 100%;
    }
    
    .concept-diagram-container {
      width: 100%;
      height: 100%;
      display: flex;
      padding: 60px 80px;
      align-items: center;
      justify-content: space-between;
      position: relative;
    }
    .left-bullets-col {
      width: 40%;
      display: flex;
      flex-direction: column;
      justify-content: center;
      z-index: 5;
    }
    .bullet-card {
      padding: 16px 20px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      margin-bottom: 16px;
      font-size: 18px;
      font-weight: 500;
      border-radius: 16px;
    }
    .mindmap-canvas {
      position: absolute;
      width: 100%;
      height: 100%;
      top: 0;
      left: 0;
      z-index: 2;
    }
    .center-node {
      position: absolute;
      width: 150px;
      height: 150px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      font-weight: 800;
      font-size: 20px;
      z-index: 10;
      padding: 18px;
      line-height: 1.2;
    }
    .leaf-node {
      position: absolute;
      padding: 14px 22px;
      background: rgba(15, 23, 42, 0.9);
      border-radius: 16px;
      font-weight: 700;
      font-size: 16px;
      text-align: center;
      box-shadow: 0 12px 28px rgba(0,0,0,0.3);
      z-index: 12;
      white-space: nowrap;
    }

    .cycle-container {
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 50px 60px;
    }
    .cycle-canvas {
      width: 400px;
      height: 400px;
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .cycle-svg {
      position: absolute;
      width: 100%;
      height: 100%;
      top: 0;
      left: 0;
      transform: rotate(-90deg);
    }
    .cycle-stage {
      position: absolute;
      width: 120px;
      height: 120px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      border-radius: 24px;
      padding: 12px;
      z-index: 20;
    }
    .cycle-stage-badge {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .cycle-stage-label {
      font-size: 14px;
      font-weight: 700;
      text-align: center;
      line-height: 1.2;
    }

    .math-container {
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 50px 60px;
    }
    .math-formula-board {
      background: rgba(15, 23, 42, 0.6);
      border-radius: 20px;
      padding: 24px 40px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 32px;
      margin-bottom: 35px;
      box-shadow: 0 12px 36px rgba(0,0,0,0.3);
    }
    .math-step-card {
      display: flex;
      align-items: center;
      border-radius: 16px;
      padding: 16px 24px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      width: 85%;
      margin-bottom: 12px;
    }
    .math-step-badge {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      color: #000000;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 14px;
      margin-right: 20px;
      flex-shrink: 0;
    }
    .math-step-text {
      font-size: 20px;
      font-weight: 600;
      width: 100%;
    }

    .comparison-container {
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 50px 60px;
    }
    .comparison-grid {
      display: flex;
      width: 90%;
      justify-content: space-between;
      gap: 40px;
      margin-top: 20px;
    }
    .comparison-column {
      width: 50%;
      border-radius: 24px;
      padding: 30px;
      box-shadow: 0 12px 32px rgba(0,0,0,0.25);
    }
    .comparison-col-header {
      font-size: 26px;
      font-weight: 800;
      margin-bottom: 24px;
      border-bottom: 2px solid;
      padding-bottom: 12px;
    }
    .comparison-bullet {
      font-size: 18px;
      font-weight: 500;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
    }
    .comparison-bullet-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      margin-right: 14px;
      flex-shrink: 0;
    }

    .timeline-container {
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 50px 60px;
    }
    .timeline-track {
      position: relative;
      width: 80%;
      height: 250px;
      display: flex;
      align-items: center;
    }
    .timeline-svg {
      position: absolute;
      left: 0;
      top: 50%;
      transform: translateY(-50%);
      width: 100%;
      height: 20px;
      pointer-events: none;
      z-index: 1;
    }
    .timeline-stage {
      position: absolute;
      top: 50%;
      transform: translate(-50%, -50%);
      display: flex;
      flex-direction: column;
      align-items: center;
      z-index: 10;
      width: 120px;
    }
    .timeline-stage-circle {
      position: relative;
      width: 80px;
      height: 80px;
      border-radius: 50%;
      background: linear-gradient(135deg, #090d16 0%, #151824 100%);
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 18px;
    }
    .timeline-stage-circle svg {
      width: 100%;
      height: 100%;
    }
    .timeline-stage-badge {
      position: absolute;
      top: -6px;
      right: -6px;
      width: 26px;
      height: 26px;
      border-radius: 50%;
      color: #000000;
      font-size: 12px;
      font-weight: 900;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 2px solid #090d16;
      box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }
    .timeline-stage-label {
      margin-top: 16px;
      font-size: 14px;
      font-weight: 700;
      color: #e2e8f0;
      text-align: center;
      line-height: 1.3;
      text-shadow: 0 2px 4px rgba(0,0,0,0.6);
    }

    .database-container {
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 50px 60px;
    }
    .database-grid-card {
      background: rgba(15, 23, 42, 0.6);
      border-radius: 24px;
      padding: 24px;
      box-shadow: 0 16px 48px rgba(0, 0, 0, 0.4);
      width: 90%;
      margin-top: 20px;
      overflow: hidden;
    }
    .database-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 16px;
    }
    .database-table th {
      text-align: left;
      padding: 16px 20px;
      font-weight: 800;
      color: rgba(255, 255, 255, 0.6);
      border-bottom: 2px solid rgba(255,255,255,0.1);
    }
    .database-table td {
      padding: 16px 20px;
      font-weight: 600;
      color: #ffffff;
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    .venn-container {
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 50px 60px;
    }
    .venn-headers {
      display: flex;
      justify-content: space-between;
      width: 600px;
      margin-bottom: 20px;
      font-size: 24px;
      font-weight: 800;
    }
    .venn-diagram-canvas {
      width: 600px;
      height: 400px;
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .venn-circle-left {
      position: absolute;
      width: 320px;
      height: 320px;
      border-radius: 50%;
      z-index: 1;
    }
    .venn-circle-right {
      position: absolute;
      width: 320px;
      height: 320px;
      border-radius: 50%;
      z-index: 1;
    }
    .venn-content-left {
      position: absolute;
      width: 180px;
      left: 30px;
      top: 80px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      z-index: 10;
    }
    .venn-content-middle {
      position: absolute;
      width: 180px;
      top: 80px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      align-items: center;
      z-index: 10;
    }
    .venn-content-right {
      position: absolute;
      width: 180px;
      right: 30px;
      top: 80px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      z-index: 10;
    }
    .venn-item-card {
      font-size: 14px;
      font-weight: 600;
      background: rgba(15, 23, 42, 0.75);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 10px;
      padding: 10px 14px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    .quiz-container {
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 50px 60px;
    }
    .quiz-card {
      background: rgba(15, 23, 42, 0.6);
      border-radius: 24px;
      padding: 36px 40px;
      width: 85%;
      margin-top: 20px;
      box-shadow: 0 16px 48px rgba(0,0,0,0.4);
    }
    .quiz-question {
      font-size: 26px;
      font-weight: 800;
      line-height: 1.3;
      margin-bottom: 30px;
    }
    .quiz-options-list {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .quiz-option {
      display: flex;
      align-items: center;
      padding: 18px 24px;
      border-radius: 16px;
      font-size: 18px;
      font-weight: 600;
      transition: all 0.3s ease;
    }
    .quiz-option-index {
      width: 30px;
      height: 30px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      margin-right: 20px;
    }
    
    .illustrated-canvas {
      width: 100%;
      height: 100%;
      position: relative;
    }
    
    .image-scene-container {
      width: 100%;
      height: 100%;
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    .scene-image {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      box-shadow: 0 12px 48px rgba(0,0,0,0.5);
      border-radius: 16px;
    }
  </style>
</head>
<body>

  <div class="composition" data-composition-id="root" data-start="0" data-width="1280" data-height="720" data-duration="${totalDuration}">
`;

  // Dynamic content injection per scene
  scenesWithTiming.forEach((scene) => {
    const sId = scene.scene_no;
    const templateId = scene.template_id || 'general_scene';
    const data = scene.template_data || {};
    
    html += `
    <!-- Scene ${sId}: ${templateId} -->
    <div class="scene clip" data-start="${scene.start}" data-duration="${scene.duration}" id="scene-${sId}">
    `;

    // 1. Render Template Layouts
    const template = templates[templateId] || templates['general_scene'];
    html += template.render(sId, data, storyboard);


    // Teacher narration subtitles display for each scene
    html += `
        <!-- Narration Subtitles banner -->
        <div class="subtitles-container" id="subtitles-${sId}">
          ${scene.teacher_script || ''}
        </div>
      </div>
    `;
  });

  // Include Audio elements for each scene (if audio exists)
  scenesWithTiming.forEach((scene) => {
    // Resolve audio from legacy field, timeline, or local scene_N.wav next to index.html
    let audioUrl = scene.audio_url
      || (scene.timeline && scene.timeline.audio_url)
      || null;
    const fallbackName = `scene_${scene.scene_no}.wav`;
    const fallbackPath = path.join(lessonDir, fallbackName);
    if (!audioUrl && fs.existsSync(fallbackPath)) {
      audioUrl = './' + fallbackName;
    }

    if (audioUrl) {
      const fileName = path.basename(audioUrl);
      const localPath = path.join(lessonDir, fileName);
      
      // If the file is not in lessonDir but exists in the project uploads folder, copy it
      if (!fs.existsSync(localPath)) {
        let uploadsPath = '';
        if (audioUrl.startsWith('/')) {
          // absolute path relative to project root
          uploadsPath = path.join(__dirname, '..', audioUrl);
        } else {
          uploadsPath = path.resolve(path.join(__dirname, '..', 'uploads', 'visual_lessons', storyboard.lesson_id || storyboard.lesson_uuid || '', fileName));
        }
        
        if (fs.existsSync(uploadsPath)) {
          fs.copyFileSync(uploadsPath, localPath);
          console.log(`Copied audio file from uploads: ${fileName}`);
        }
      }
      
      // Determine final audio src path (relative to index.html in lessonDir)
      let audioPath = audioUrl;
      if (fs.existsSync(localPath)) {
        audioPath = './' + fileName;
      } else if (audioPath.startsWith('/')) {
        audioPath = '.' + audioPath;
      } else {
        console.warn(`[Audio Warning] Missing audio file for scene ${scene.scene_no}: ${audioPath}`);
      }
      
      html += `
      <audio class="clip" data-start="${scene.start}" data-duration="${scene.duration}" src="${audioPath}" id="audio-scene-${scene.scene_no}"></audio>
      `;
    }
  });

  html += `
  </div>

  <script>
    // active theme variables loaded dynamically
    const storyboardThemeName = "${storyboard.theme || 'indigo'}";
    const theme = window.getTheme(storyboardThemeName);
    
    // Apply styling rules
    document.body.style.background = theme.background;
    document.body.style.fontFamily = theme.fontFamily;
    document.body.style.color = theme.textColor;
    
    // Dynamic Styling Updates across all template headers
    document.querySelectorAll('.theme-text').forEach(el => {
      el.style.color = theme.accentColor;
    });
    document.querySelectorAll('.theme-stroke').forEach(el => {
      el.style.stroke = theme.accentColor;
    });
    document.querySelectorAll('.theme-fill').forEach(el => {
      el.style.fill = theme.accentColor;
    });
    document.querySelectorAll('.theme-accent-bg').forEach(el => {
      el.style.background = theme.accentColor;
    });
    document.querySelectorAll('.theme-card-bg').forEach(el => {
      el.style.background = theme.cardBackground;
    });
    document.querySelectorAll('.theme-card-border').forEach(el => {
      el.style.border = theme.cardBorder;
    });
    document.querySelectorAll('.theme-accent-border').forEach(el => {
      el.style.borderColor = theme.accentColor;
    });

    // Populate dynamic LaTeX inputs in DOM
    const rawData = ${JSON.stringify(scenesWithTiming)};
    
    rawData.forEach(scene => {
      const sId = scene.scene_no;
      const data = scene.template_data || {};
      
      if (scene.template_id === 'math_derivation') {
        const fEl = document.getElementById('math-formula-' + sId);
        if (fEl && data.formula) {
          fEl.innerHTML = katex.renderToString(data.formula.trim().replace(/^\\$+|\\$+$/g, ''), { throwOnError: false });
        }
        // Prefer normalized steps; fall back to equation_steps from LLM payloads
        const mathSteps = (data.steps && data.steps.length)
          ? data.steps
          : (data.equation_steps || []).map(function(s) {
              if (typeof s === 'string') return s;
              var label = (s && (s.step || s.label)) || '';
              var value = (s && (s.value || s.result)) || '';
              return label && value ? (label + ': ' + value) : (label || value || '');
            });
        mathSteps.forEach((step, sIdx) => {
          const sEl = document.getElementById('math-step-text-' + sId + '-' + sIdx);
          if (sEl && typeof step === 'string' && step.trim()) {
            try {
              sEl.innerHTML = katex.renderToString(step.trim().replace(/^\\$+|\\$+$/g, ''), { throwOnError: false });
            } catch (e) {
              sEl.textContent = step;
            }
          }
        });
      }
    });

    // Center and scale illustrated SVGs dynamically
    rawData.forEach(scene => {
      const sId = scene.scene_no;
      if (scene.template_id === 'illustrated_scene') {
        const group = document.getElementById('ill-group-' + sId);
        if (group) {
          const bbox = group.getBBox();
          if (bbox.width > 0 && bbox.height > 0) {
            const targetWidth = 1280 * 0.70;
            const targetHeight = 720 * 0.70;
            const scaleX = targetWidth / bbox.width;
            const scaleY = targetHeight / bbox.height;
            const scale = Math.min(2.0, Math.min(scaleX, scaleY));
            const dx = 640 - (bbox.x + bbox.width / 2) * scale;
            const dy = 360 - (bbox.y + bbox.height / 2) * scale;
            group.setAttribute("transform", 'translate(' + dx + ', ' + dy + ') scale(' + scale + ')');
          }
        }
      }
    });

    // Build the master timeline
    const mainTimeline = gsap.timeline();
  `;

  // Stitch all animations sequentially using main GSAP timeline
  const AnimationAdapter = require('./engine/animation/adapters/AnimationAdapter');
  const AnimationController = require('./engine/animation/controllers/AnimationController');
  const GSAPGenerator = require('./engine/animation/generator/GSAPGenerator');

  scenesWithTiming.forEach(scene => {
    const sId = scene.scene_no;
    const templateId = scene.template_id || 'general_scene';
    const data = scene.template_data || {};
    const template = templates[templateId] || templates['general_scene'];
    
    const animResult = template.animate ? template.animate(sId, data, null, scene.duration) : '';
    const sceneTimeline = typeof animResult === 'string'
      ? AnimationAdapter.adaptLegacy(animResult)
      : animResult;

    // Resolve entry transition
    let transitionScript = '';
    if (scene.scene_no !== 1) {
      const TransitionEngine = require('./engine/theme/transitions/TransitionEngine');
      const TransitionTheme = require('./engine/theme/manager/ThemeManager').getTheme(scene.themeId);
      const transitionStyle = TransitionTheme.transitionStyle || 'FADE';
      
      transitionScript = TransitionEngine.generateTransition(transitionStyle, `#scene-${sId}`, 0.4);
    }

    let compiledScript = '';
    if (sceneTimeline) {
      const controller = new AnimationController(sceneTimeline);
      controller.validate();
      compiledScript = GSAPGenerator.generateGSAP(sceneTimeline);
    }

    // Resolve dynamic camera motion for scene
    let cameraScript = '';
    const camObj = scene.camera || (scene.metadata && scene.metadata.camera);
    if (camObj) {
      const camZoom = camObj.zoom || 1.0;
      const camPanX = camObj.pan_x || 0;
      const camPanY = camObj.pan_y || 0;
      if (camZoom !== 1.0 || camPanX !== 0 || camPanY !== 0) {
        cameraScript = `
      sceneTl.fromTo('#scene-${sId} .camera-viewport-wrapper',
        { scale: 1.0, x: 0, y: 0 },
        { scale: ${camZoom}, x: ${camPanX}, y: ${camPanY}, duration: ${(scene.duration || 5.0).toFixed(1)}, ease: 'power1.inOut' },
        0
      );
`;
      }
    }

    compiledScript = transitionScript + cameraScript + compiledScript;
    
    html += `
    // Scene ${sId} (${templateId}) Animation Timeline
    {
      const sceneTl = gsap.timeline();
${compiledScript}      mainTimeline.add(sceneTl, ${scene.start});
    }
    `;
  });

  html += `
    window.__timelines = window.__timelines || {};
    window.__timelines["root"] = mainTimeline;

    // ════════════════════════════════════════════════════════════════════════
    // HYPERFRAMES AUTHORITATIVE PLAYBACK ENGINE & IPC MESSAGING PROTOCOL
    // ════════════════════════════════════════════════════════════════════════
    window.HyperframesEngine = {
      scenes: rawData,
      totalDuration: ${totalDuration},
      currentSceneNo: 1,
      isPlaying: false,
      isMuted: false,
      speed: 1.0,
      lastEmittedTime: -1,

      init() {
        console.log("🚀 [Hyperframes Runtime] Initializing authoritative playback engine...");
        
        // Main GSAP timeline starts paused - waiting for parent UI command or auto-start
        mainTimeline.pause();
        this.updateSceneVisibility(1);

        // Frame ticker sync loop for time, scene, and audio updates
        gsap.ticker.add(() => {
          if (!this.isPlaying) return;
          const curTime = mainTimeline.time();
          this.syncStateForTime(curTime);

          // Throttle CURRENT_TIME events to 10Hz to avoid postMessage flood
          if (Math.abs(curTime - this.lastEmittedTime) >= 0.1) {
            this.lastEmittedTime = curTime;
            this.emit('CURRENT_TIME', { currentTime: curTime, duration: this.totalDuration });
          }

          if (curTime >= this.totalDuration) {
            this.pause();
            this.emit('TIMELINE_FINISHED');
          }
        });

        // Notify parent window that Hyperframes Runtime is ready with full metadata
        this.emit('READY', {
          totalScenes: this.scenes.length,
          duration: this.totalDuration,
          lessonTitle: "${storyboard.lesson_title || 'Visual Storyboard Video'}",
          scenes: this.scenes.map(s => ({
            scene_no: s.scene_no,
            title: s.title || ("Scene " + s.scene_no),
            teacher_script: s.teacher_script || "",
            start: s.start,
            duration: s.duration
          }))
        });

        // Auto-start timeline playback by default
        setTimeout(() => {
          if (!this.isPlaying) {
            console.log("▶ [Hyperframes Engine] Auto-starting video composition playback.");
            this.play();
          }
        }, 150);
      },

      updateSceneVisibility(sceneNo) {
        this.currentSceneNo = sceneNo;
        this.scenes.forEach(s => {
          const el = document.getElementById('scene-' + s.scene_no);
          if (el) {
            if (s.scene_no === sceneNo) {
              el.style.display = 'block';
              el.style.visibility = 'visible';
              el.style.opacity = '1';
            } else {
              el.style.display = 'none';
              el.style.visibility = 'hidden';
              el.style.opacity = '0';
            }
          }
        });
      },

      syncStateForTime(time) {
        let activeScene = this.scenes[0];
        for (let i = 0; i < this.scenes.length; i++) {
          const s = this.scenes[i];
          const nextS = this.scenes[i + 1];
          if (time >= s.start && (!nextS || time < nextS.start)) {
            activeScene = s;
            break;
          }
        }

        if (activeScene && activeScene.scene_no !== this.currentSceneNo) {
          this.updateSceneVisibility(activeScene.scene_no);
          const sceneOffset = time - activeScene.start;
          this.syncAudioForScene(activeScene.scene_no, sceneOffset);

          this.emit('SCENE_CHANGED', {
            currentScene: activeScene.scene_no,
            totalScenes: this.scenes.length,
            title: activeScene.title,
            script: activeScene.teacher_script
          });
          this.emit('SUBTITLE_CHANGED', { script: activeScene.teacher_script });
        }
      },

      syncAudioForScene(sceneNo, offset) {
        document.querySelectorAll('audio').forEach(a => a.pause());
        const activeAudio = document.getElementById('audio-scene-' + sceneNo);
        if (activeAudio) {
          try {
            activeAudio.currentTime = Math.max(0, offset);
            activeAudio.muted = this.isMuted;
            activeAudio.playbackRate = this.speed;
            if (this.isPlaying) {
              activeAudio.play().catch(e => {});
            }
          } catch(e) {}
        }
      },

      play() {
        this.isPlaying = true;
        mainTimeline.play();
        mainTimeline.timeScale(this.speed);
        const curTime = mainTimeline.time();
        this.syncStateForTime(curTime);

        const activeAudio = document.getElementById('audio-scene-' + this.currentSceneNo);
        if (activeAudio) {
          try {
            activeAudio.muted = this.isMuted;
            activeAudio.playbackRate = this.speed;
            activeAudio.play().catch(e => {});
          } catch(e) {}
        }
        this.emit('PLAYING');
      },

      pause() {
        this.isPlaying = false;
        mainTimeline.pause();
        document.querySelectorAll('audio').forEach(a => a.pause());
        this.emit('PAUSED');
      },

      seek(targetTime) {
        const clamped = Math.max(0, Math.min(targetTime, this.totalDuration));
        mainTimeline.seek(clamped);
        this.syncStateForTime(clamped);
        this.emit('CURRENT_TIME', { currentTime: clamped, duration: this.totalDuration });
      },

      setSpeed(rate) {
        this.speed = rate;
        mainTimeline.timeScale(rate);
        document.querySelectorAll('audio').forEach(a => a.playbackRate = rate);
      },

      setMute(isMuted) {
        this.isMuted = isMuted;
        document.querySelectorAll('audio').forEach(a => a.muted = isMuted);
      },

      jumpToScene(sceneNo) {
        const s = this.scenes.find(sc => sc.scene_no === sceneNo);
        if (s) {
          this.seek(s.start);
          if (this.isPlaying) this.play();
        }
      },

      emit(type, payload = {}) {
        try {
          window.parent.postMessage({ source: 'HYPERFRAMES_ENGINE', type, ...payload }, '*');
        } catch(e) {
          console.error("Failed to emit postMessage from Hyperframes Engine:", e);
        }
      }
    };

    // Incoming Command Listener from Parent UI
    window.addEventListener('message', (e) => {
      const data = e.data;
      if (!data || data.target !== 'HYPERFRAMES_ENGINE') return;
      console.log("📥 [Hyperframes Engine Command Received]:", data.command, data);
      
      switch(data.command) {
        case 'PLAY': window.HyperframesEngine.play(); break;
        case 'PAUSE': window.HyperframesEngine.pause(); break;
        case 'SEEK': window.HyperframesEngine.seek(data.targetTime); break;
        case 'RESTART': window.HyperframesEngine.seek(0); window.HyperframesEngine.play(); break;
        case 'SET_PLAYBACK_RATE': window.HyperframesEngine.setSpeed(data.rate); break;
        case 'TOGGLE_MUTE': window.HyperframesEngine.setMute(data.isMuted); break;
        case 'JUMP_SCENE': window.HyperframesEngine.jumpToScene(data.sceneNo); break;
      }
    });

    // Auto-initialize when DOM is ready
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
      window.HyperframesEngine.init();
    } else {
      window.addEventListener('DOMContentLoaded', () => window.HyperframesEngine.init());
    }
  </script>
</body>
</html>
`;

  const masterPath = path.join(lessonDir, 'index.html');
  fs.writeFileSync(masterPath, html, 'utf8');

  const profilerRecord = PerformanceAPI.getProfiler().stop('generate_html');
  const htmlDuration = MetricsCollector.stop('generate_html');
  Logger.info('HTML generation pipeline complete', { outputPath: masterPath, durationMs: htmlDuration, profilerMs: profilerRecord.durationMs, heapUsedMb: profilerRecord.heapUsedMb });
  MetricsCollector.record('html_output_path', masterPath);
  MetricsCollector.record('scene_count', scenes.length);

  // Cache the compiled HTML path for this lesson for repeat lookups
  const lessonId = storyboard.lesson_id || path.basename(lessonDir);
  PerformanceAPI.getCache('render').set(`html:${lessonId}`, masterPath);

  console.log(`\nGenerated master video composition HTML file at:`);
  console.log(masterPath);
  callback(masterPath);
}

main();
