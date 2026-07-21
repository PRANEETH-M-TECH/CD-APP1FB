/**
 * PedagogicalStrategyEngine.js
 * Controls pedagogical concept sequencing rules, progressive disclosure schedules,
 * and reinforcement recaps.
 */
class PedagogicalStrategyEngine {
  /**
   * Applies the selected pedagogical strategy to expand and sequence instruction steps.
   * @param {Array<TeachingStep>} steps 
   * @param {PedagogicalStrategy} strategy 
   * @returns {Array<TeachingStep>}
   */
  static processSequence(steps, strategy) {
    const s = strategy || { reinforcement_rules: [] };

    // 1. Concept Sequencing: Prerequisite priority ordering
    const sorted = [...steps].sort((a, b) => {
      const aPri = (a.metadata && a.metadata.priority) || 0;
      const bPri = (b.metadata && b.metadata.priority) || 0;
      return aPri - bPri;
    });

    // 2. Reinforcement rules: Inject visual recaps or reviews
    const enriched = [];
    sorted.forEach((step, idx) => {
      enriched.push(step);

      // Inject recap step if requested
      if (idx > 0 && idx % 2 === 1 && s.reinforcement_rules && s.reinforcement_rules.includes('auto_recap')) {
        const TeachingStep = require('../../teaching/models/TeachingStep');
        enriched.push(new TeachingStep({
          action_type: 'REVIEW',
          target_id: step.targetId,
          duration: 3,
          script: 'Let us take a brief moment to review what we just covered.',
          metadata: { reinforcement: true }
        }));
      }
    });

    return enriched;
  }
}

module.exports = PedagogicalStrategyEngine;
