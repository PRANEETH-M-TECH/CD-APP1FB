const Scene = require('../engine/scene/Scene');
const Renderer = require('../engine/renderer/Renderer');

/**
 * QuizCheckpoint.js
 * Template orchestrator delegating layout rendering entirely to the engine Renderer.
 */
module.exports = {
  render: (sId, data, storyboard) => {
    const sceneJson = storyboard.scenes.find(s => s.scene_no === sId);
    const scene = Scene.deserialize(sceneJson);
    return Renderer.renderScene(scene);
  },
  animate: (sId, data) => {
    return `
      sceneTl.fromTo('#quiz-card-${sId}', { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out' });
      sceneTl.fromTo('#quiz-options-${sId} .quiz-option', { opacity: 0, x: -20 }, { opacity: 1, x: 0, stagger: 0.15, duration: 0.4 }, 0.3);
      
      // Resolve correct answer index dynamically (supports is_correct, index, id, or answer string)
      const optionsData_${sId} = ${JSON.stringify(data.options || [])};
      let correctIdx_${sId} = 0;
      if (optionsData_${sId}.length > 0) {
        if (typeof optionsData_${sId}[0] === 'object') {
          correctIdx_${sId} = optionsData_${sId}.findIndex(opt => opt.is_correct);
          if (correctIdx_${sId} === -1 && ${JSON.stringify(data.correct_answer_id || null)} != null) {
            const wantId = ${JSON.stringify(data.correct_answer_id || null)};
            correctIdx_${sId} = optionsData_${sId}.findIndex(opt => String(opt.id) === String(wantId));
          }
          if (correctIdx_${sId} === -1) correctIdx_${sId} = 0;
        } else {
          const idxAlias = ${data.correct_answer_index != null ? data.correct_answer_index : (data.correct_option_index != null ? data.correct_option_index : (data.correct_idx != null ? data.correct_idx : 'null'))};
          if (idxAlias !== null) {
            correctIdx_${sId} = idxAlias;
          } else if (${JSON.stringify(data.correct_answer || null)} != null) {
            const ans = String(${JSON.stringify(data.correct_answer || '')}).trim().toLowerCase();
            correctIdx_${sId} = optionsData_${sId}.findIndex(opt => {
              const t = String(opt).trim().toLowerCase();
              return t === ans || t.startsWith(ans) || ans.includes(t);
            });
            if (correctIdx_${sId} === -1) correctIdx_${sId} = 0;
          }
        }
      }
      
      const correctOpt_${sId} = document.getElementById('quiz-opt-${sId}-' + correctIdx_${sId});
      if (correctOpt_${sId}) {
        sceneTl.to('#quiz-options-${sId} .quiz-option', { opacity: 0.3, duration: 0.4 }, 1.2);
        sceneTl.to(correctOpt_${sId}, { 
          opacity: 1, 
          borderColor: '#22c55e', 
          background: 'rgba(34, 197, 94, 0.15)',
          boxShadow: '0 0 24px rgba(34, 197, 94, 0.4)',
          duration: 0.5 
        }, 1.4);
      }
    `;
  }
};
