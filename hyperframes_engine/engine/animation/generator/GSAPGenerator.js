/**
 * GSAPGenerator.js
 * The single generation authority compiling serializable Animation timelines
 * into raw GSAP JavaScript instructions.
 */
class GSAPGenerator {
  /**
   * Compiles an AnimationTimeline into GSAP timeline code strings.
   * @param {AnimationTimeline} timeline 
   * @returns {string} Compiled GSAP timeline commands
   */
  static generateGSAP(timeline) {
    if (!timeline || !Array.isArray(timeline.animations)) {
      return '';
    }

    let script = '';
    timeline.animations.forEach((anim) => {
      const { type, target, duration, delay, easing, repeat, metadata } = anim;

      // Map targets to class/ID selector strings
      const targetSelector = typeof target === 'string' && (target.startsWith('#') || target.startsWith('.') || target.startsWith('['))
        ? target
        : `#${target}`;

      const easeVal = easing ? `, ease: '${easing}'` : '';
      const repeatVal = repeat ? `, repeat: ${repeat}` : '';
      const delayVal = delay ? `, delay: ${delay}` : '';

      switch (type.toUpperCase()) {
        case 'FADE_IN':
          script += `      sceneTl.fromTo('${targetSelector}', { opacity: 0 }, { opacity: 1, duration: ${duration}${easeVal}${repeatVal}${delayVal} });\n`;
          break;

        case 'FADE_OUT':
          script += `      sceneTl.fromTo('${targetSelector}', { opacity: 1 }, { opacity: 0, duration: ${duration}${easeVal}${repeatVal}${delayVal} });\n`;
          break;

        case 'SCALE': {
          const fromScale = metadata.scaleStart !== undefined ? metadata.scaleStart : 0;
          const toScale = metadata.scaleEnd !== undefined ? metadata.scaleEnd : 1;
          script += `      sceneTl.fromTo('${targetSelector}', { scale: ${fromScale} }, { scale: ${toScale}, duration: ${duration}${easeVal}${repeatVal}${delayVal} });\n`;
          break;
        }

        case 'MOVE': {
          const fromX = metadata.fromX !== undefined ? metadata.fromX : 0;
          const toX = metadata.toX !== undefined ? metadata.toX : 0;
          const fromY = metadata.fromY !== undefined ? metadata.fromY : 0;
          const toY = metadata.toY !== undefined ? metadata.toY : 0;
          script += `      sceneTl.fromTo('${targetSelector}', { x: ${fromX}, y: ${fromY} }, { x: ${toX}, y: ${toY}, duration: ${duration}${easeVal}${repeatVal}${delayVal} });\n`;
          break;
        }

        case 'ROTATE': {
          const fromRot = metadata.fromRotation !== undefined ? metadata.fromRotation : 0;
          const toRot = metadata.toRotation !== undefined ? metadata.toRotation : 360;
          script += `      sceneTl.fromTo('${targetSelector}', { rotation: ${fromRot} }, { rotation: ${toRot}, duration: ${duration}${easeVal}${repeatVal}${delayVal} });\n`;
          break;
        }

        case 'CUSTOM':
          script += `      ${metadata.script || ''}\n`;
          break;
      }
    });

    return script;
  }
}

module.exports = GSAPGenerator;
