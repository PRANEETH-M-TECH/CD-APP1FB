const fs = require('fs');
const path = require('path');

/**
 * Diagnostics.js
 * Startup health checks, dependency verification, and engine diagnostics.
 * Provides developer-friendly diagnostic output.
 */
class Diagnostics {
  /**
   * Runs all startup diagnostic checks and returns a health report.
   * @param {object} config  Active HyperframesConfig.get() result
   * @returns {{ healthy: boolean, checks: object[] }}
   */
  static report(config) {
    const checks = [];

    // 1. Node.js version check
    const nodeVer = process.versions.node;
    checks.push({
      name: 'node_version',
      status: 'ok',
      detail: `Node.js ${nodeVer}`
    });

    // 2. Engine module presence checks
    const engineDir = path.join(__dirname, '../..');
    const expectedModules = [
      'scene', 'components', 'assets', 'camera', 'layout',
      'animation', 'focus', 'theme', 'teaching', 'planner',
      'synchronization', 'subtitles', 'pedagogy', 'renderer', 'adapters'
    ];
    expectedModules.forEach(mod => {
      const exists = fs.existsSync(path.join(engineDir, mod));
      checks.push({
        name: `engine_module_${mod}`,
        status: exists ? 'ok' : 'missing',
        detail: exists ? `${mod}/ found` : `${mod}/ NOT found`
      });
    });

    // 3. Config validity
    const configValid = config && config.renderer && config.engine;
    checks.push({
      name: 'config',
      status: configValid ? 'ok' : 'invalid',
      detail: configValid ? `env=${config.env}, logLevel=${config.logLevel}` : 'Config object is incomplete'
    });

    // 4. Output directory presence
    const outputDir = path.join(__dirname, '../../../outputs');
    checks.push({
      name: 'outputs_directory',
      status: fs.existsSync(outputDir) ? 'ok' : 'warn',
      detail: fs.existsSync(outputDir) ? 'outputs/ exists' : 'outputs/ not found — will be created on first render'
    });

    const failing = checks.filter(c => c.status === 'missing' || c.status === 'invalid');
    return {
      healthy: failing.length === 0,
      checks,
      summary: failing.length === 0
        ? 'All Hyperframes engine diagnostics passed.'
        : `${failing.length} diagnostic check(s) failed: ${failing.map(f => f.name).join(', ')}`
    };
  }
}

module.exports = Diagnostics;
