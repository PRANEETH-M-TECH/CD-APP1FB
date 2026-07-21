const HyperframesError = require('../errors/HyperframesError');

/**
 * ValidationFramework.js
 * Centralized validation runner. Accepts a subject and a list of validator
 * functions, aggregating results into a structured report.
 */
class ValidationFramework {
  /**
   * Runs a list of validator functions against a subject.
   * Each validator should return { isValid: bool, errors: [], warnings: [] }
   * @param {*} subject 
   * @param {Array<function>} validators 
   * @param {string} stageName 
   * @returns {{ isValid: boolean, errors: string[], warnings: string[] }}
   */
  static run(subject, validators = [], stageName = 'unknown') {
    const errors = [];
    const warnings = [];

    for (const validator of validators) {
      try {
        const result = validator(subject);
        if (result) {
          if (Array.isArray(result.errors)) errors.push(...result.errors);
          if (Array.isArray(result.warnings)) warnings.push(...result.warnings);
        }
      } catch (err) {
        errors.push(`[${stageName}] Validator threw unexpected error: ${err.message}`);
      }
    }

    return { isValid: errors.length === 0, errors, warnings, stage: stageName };
  }

  /**
   * Wraps existing StoryboardValidator for unified usage.
   * @param {object} storyboardJson 
   * @returns {{ isValid: boolean, errors: string[], warnings: string[] }}
   */
  static validateStoryboard(storyboardJson) {
    try {
      const StoryboardValidator = require('../../planner/validators/StoryboardValidator');
      const result = StoryboardValidator.validate(storyboardJson);
      return {
        isValid: result.isValid,
        errors: result.errors || [],
        warnings: [],
        stage: 'storyboard'
      };
    } catch (err) {
      return {
        isValid: false,
        errors: [`StoryboardValidator threw: ${err.message}`],
        warnings: [],
        stage: 'storyboard'
      };
    }
  }

  /**
   * Asserts validation passes — throws a HyperframesError on critical failure.
   * @param {object} result 
   */
  static assertValid(result) {
    if (!result.isValid) {
      throw HyperframesError.pipeline(result.stage, result.errors.join('; '));
    }
  }
}

module.exports = ValidationFramework;
