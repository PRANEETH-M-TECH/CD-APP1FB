const { spawn } = require('child_process');
const path = require('path');

// Execute the selection script inside remotion_test_app
const scriptPath = path.join(__dirname, 'remotion_test_app', 'run-storyboard.js');
const child = spawn('node', [scriptPath], { stdio: 'inherit', shell: true });

child.on('close', (code) => {
  process.exit(code);
});
