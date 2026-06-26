const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { spawn } = require('child_process');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const uploadsDir = path.join(__dirname, 'public', 'uploads', 'visual_lessons');

function main() {
  console.log("=========================================");
  console.log("   REMOTION VISUAL STORYBOARD SELECTOR   ");
  console.log("=========================================");

  if (!fs.existsSync(uploadsDir)) {
    console.log("No storyboards folder found yet at:");
    console.log(uploadsDir);
    console.log("\nGenerate a visual lesson storyboard first from the app or CLI helper!");
    rl.close();
    return;
  }

  const dirs = fs.readdirSync(uploadsDir).filter(f => {
    return fs.statSync(path.join(uploadsDir, f)).isDirectory();
  });

  if (dirs.length === 0) {
    console.log("No generated storyboards found in:");
    console.log(uploadsDir);
    console.log("\nGenerate a visual lesson storyboard first!");
    rl.close();
    return;
  }

  const lessons = [];
  dirs.forEach((dir) => {
    const jsonPath = path.join(uploadsDir, dir, 'lesson.json');
    if (fs.existsSync(jsonPath)) {
      try {
        const data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
        lessons.push({
          id: dir,
          title: data.lesson_title || "Unnamed Lesson",
          theme: data.theme || "indigo",
          scenesCount: data.scenes ? data.scenes.length : 0
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
    console.log(`\nYou selected: ${selectedLesson.title} (${selectedLesson.id})`);
    console.log("\nChoose Action:");
    console.log("[1] Preview on Localhost (Browser Player)");
    console.log("[2] Render/Export to MP4 Video File");

    rl.question(`Select action (1-2): `, (actionAnswer) => {
      const action = actionAnswer.trim();
      const propsPath = `public/uploads/visual_lessons/${selectedLesson.id}/lesson.json`;

      let cmd = 'npx';
      let args = [];

      if (action === '1') {
        console.log(`\nLaunching Remotion Player on localhost...`);
        args = ['remotion', 'preview', 'src/index.ts', `--props=${propsPath}`];
      } else if (action === '2') {
        const outName = `output_${selectedLesson.id}.mp4`;
        console.log(`\nRendering video to ${outName} (this might take a minute)...`);
        args = ['remotion', 'render', 'src/index.ts', 'StoryboardVideo', outName, `--props=${propsPath}`];
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
  });
}

main();
