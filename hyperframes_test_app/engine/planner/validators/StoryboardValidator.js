/**
 * StoryboardValidator.js
 * ValidatesStoryboard JSON schemas for educational coverage:
 * - Every learning objective must map to teaching steps.
 * - Every teaching step must coordinate with at least one scene.
 * - Every scene must target a valid teaching_step_id.
 * - Required metadata keys exist.
 */
class StoryboardValidator {
  /**
   * Validates a storyboard JSON object and returns all compilation validation errors.
   * @param {object} storyboardJson 
   * @returns {object} { isValid: boolean, errors: Array<string> }
   */
  static validate(storyboardJson) {
    const errors = [];

    if (!storyboardJson) {
      return { isValid: false, errors: ['Storyboard JSON structure is null or undefined.'] };
    }

    // Required root parameters check
    const requiredRoot = ['lesson_title', 'scenes'];
    requiredRoot.forEach(prop => {
      if (!storyboardJson[prop]) {
        errors.push(`Missing required root property: ${prop}`);
      }
    });

    const scenes = storyboardJson.scenes || [];
    const teachingPlan = storyboardJson.teaching_plan || { teaching_steps: [] };
    const steps = teachingPlan.teaching_steps || [];
    const objectives = storyboardJson.learning_objectives || [];

    // 1. Every learning objective has teaching steps
    if (objectives.length > 0 && steps.length === 0) {
      errors.push('Learning objectives are defined, but the teaching plan contains no teaching steps.');
    }

    // 2. Every teaching step has at least one scene
    steps.forEach(step => {
      const sceneUsingStep = scenes.find(s => s.teaching_step_id === step.step_id);
      if (!sceneUsingStep) {
        errors.push(`Teaching step '${step.step_id}' is not referenced by any storyboard scene.`);
      }
    });

    // 3. Every scene belongs to a teaching step
    scenes.forEach(scene => {
      if (!scene.teaching_step_id) {
        errors.push(`Scene ${scene.scene_no} does not reference a teaching_step_id.`);
      } else {
        const stepExists = steps.find(s => s.step_id === scene.teaching_step_id);
        if (!stepExists) {
          errors.push(`Scene ${scene.scene_no} references unknown teaching_step_id: '${scene.teaching_step_id}'.`);
        }
      }

      // 4. Required metadata exists
      if (!scene.metadata) {
        errors.push(`Scene ${scene.scene_no} is missing metadata.`);
      } else {
        const reqMetadata = ['learning_objective', 'difficulty'];
        reqMetadata.forEach(metaKey => {
          if (!scene.metadata[metaKey]) {
            errors.push(`Scene ${scene.scene_no} metadata is missing: '${metaKey}'.`);
          }
        });
      }
    });

    return {
      isValid: errors.length === 0,
      errors: errors
    };
  }
}

module.exports = StoryboardValidator;
