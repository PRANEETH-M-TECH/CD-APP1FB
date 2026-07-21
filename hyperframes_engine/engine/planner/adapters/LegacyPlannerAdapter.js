/**
 * LegacyPlannerAdapter.js
 * Adapts legacy storyboard JSON objects at load time, automatically populating
 * missing learning objectives, teaching plans, steps, and scene metadata.
 */
class LegacyPlannerAdapter {
  /**
   * Adapts legacy storyboards to the new educational planning schema.
   * @param {object} legacyJson 
   * @returns {object} Adapted JSON storyboard
   */
  static adapt(legacyJson) {
    if (!legacyJson) return null;

    const title = legacyJson.lesson_title || legacyJson.title || 'Legacy Lesson';

    // 1. Augment root pedagogical properties
    if (!legacyJson.learning_objectives) {
      legacyJson.learning_objectives = [title];
    }
    if (!legacyJson.student_level) {
      legacyJson.student_level = 'General';
    }
    if (!legacyJson.difficulty) {
      legacyJson.difficulty = 'intermediate';
    }
    if (!legacyJson.subject) {
      legacyJson.subject = 'general';
    }
    if (!legacyJson.concepts) {
      legacyJson.concepts = [title];
    }
    if (!legacyJson.summary) {
      legacyJson.summary = `Legacy lesson review of ${title}.`;
    }

    // 2. Build sequential teaching steps
    const scenes = legacyJson.scenes || [];
    const steps = [];

    scenes.forEach((scene) => {
      const sId = scene.scene_no;
      const stepId = `step_scene_${sId}`;

      // Assign mapping step reference to scene
      if (!scene.teaching_step_id) {
        scene.teaching_step_id = stepId;
      }

      // Add default metadata if missing
      if (!scene.metadata) {
        scene.metadata = {
          learning_objective: title,
          concept_importance: 'medium',
          instruction_type: 'EXPLAIN',
          difficulty: 'intermediate',
          teaching_strategy: 'sequential',
          reinforcement_level: 'medium',
          quiz_hints: '',
          analytics_tags: ''
        };
      }

      // Register step
      steps.push({
        step_id: stepId,
        action_type: 'EXPLAIN',
        instructional_purpose: `Explain concepts in scene ${sId}`,
        visual_purpose: `Visualize scene ${sId}`,
        concept: title,
        expected_outcome: `Understand step ${sId}`,
        related_components: [],
        duration: scene.durationInFrames ? scene.durationInFrames / 30 : 6.0,
        teaching_notes: '',
        narration_hints: scene.teacher_script || ''
      });
    });

    if (!legacyJson.teaching_plan) {
      legacyJson.teaching_plan = {
        teaching_steps: steps
      };
    }

    return legacyJson;
  }
}

module.exports = LegacyPlannerAdapter;
