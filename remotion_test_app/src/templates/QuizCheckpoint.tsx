import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, spring } from 'remotion';
import { getTheme } from '../themeHelper';

interface QuizCheckpointProps {
  question?: string;
  options?: string[];
  correct_idx?: number;
  theme: string;
  left_title?: string;
  left_bullets?: string[];
  right_column?: {
    header: string;
    bullets: string[];
  };
}

export const QuizCheckpoint: React.FC<QuizCheckpointProps> = ({
  question = '',
  options = [],
  correct_idx = 0,
  theme,
  left_title = '',
  left_bullets = [],
  right_column,
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const activeTheme = getTheme(theme);

  // Backward compatibility fallback mapping
  const finalQuestion = question || (left_bullets && left_bullets[0]) || 'Recall Checkpoint';
  let finalOptions = options || [];
  let finalCorrectIdx = correct_idx ?? 0;

  if (finalOptions.length === 0 && right_column?.bullets && right_column.bullets.length > 0) {
    finalOptions = [right_column.bullets[0]];
    finalCorrectIdx = 0;
  }

  // Question slide-down
  const questionSpring = spring({
    frame,
    fps,
    config: { stiffness: 100, damping: 14 }
  });
  const questionTranslateY = interpolate(questionSpring, [0, 1], [-40, 0]);
  const questionOpacity = questionSpring;

  // Options staggered entry
  const optionSprings = finalOptions.map((_, idx) => {
    return spring({
      frame: frame - (15 + idx * 8),
      fps,
      config: { stiffness: 120, damping: 14 }
    });
  });

  // Highlight reveal timing (reveal correct choice in last 2.5s)
  const revealFrameStart = Math.max(45, durationInFrames - 75);
  const correctSpring = spring({
    frame: frame - revealFrameStart,
    fps,
    config: { stiffness: 100, damping: 12 }
  });

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: '50px 60px',
        boxSizing: 'border-box',
        fontFamily: activeTheme.fontFamily,
        color: activeTheme.textColor,
        justifyContent: 'center',
      }}
    >
      {/* Checkpoint Header */}
      <div
        style={{
          fontSize: '14px',
          fontWeight: 800,
          textTransform: 'uppercase',
          color: activeTheme.accentColor,
          letterSpacing: '2px',
          marginBottom: '15px',
        }}
      >
        💡 Active Recall Checkpoint
      </div>

      {/* Question Card */}
      <div
        style={{
          width: '85%',
          background: 'rgba(15, 23, 42, 0.5)',
          border: activeTheme.cardBorder,
          borderRadius: '20px',
          padding: '24px 30px',
          fontSize: '24px',
          fontWeight: 700,
          textAlign: 'center',
          lineHeight: '1.4',
          marginBottom: '35px',
          boxShadow: '0 12px 32px rgba(0,0,0,0.2)',
          transform: `scale(${questionSpring}) translateY(${questionTranslateY}px)`,
          opacity: questionOpacity,
        }}
      >
        {finalQuestion}
      </div>

      {/* Options Grid Layout */}
      <div
        style={{
          width: '85%',
          display: 'grid',
          gridTemplateColumns: finalOptions.length === 1 ? '1fr' : '1fr 1fr',
          gap: '16px',
          justifyContent: 'center',
          maxWidth: finalOptions.length === 1 ? '500px' : 'none',
        }}
      >
        {finalOptions.map((opt, idx) => {
          const scale = optionSprings[idx];
          const opacity = optionSprings[idx];
          const isCorrect = idx === finalCorrectIdx;

          // Reveal animations styling
          let cardBg = activeTheme.cardBackground;
          let borderStyle = activeTheme.cardBorder;
          let textColor = activeTheme.textColor;

          if (frame >= revealFrameStart) {
            if (isCorrect) {
              const alpha = interpolate(correctSpring, [0, 1], [0.05, 0.15]);
              cardBg = `rgba(16, 185, 129, ${alpha})`;
              borderStyle = `2.5px solid rgba(16, 185, 129, ${interpolate(correctSpring, [0, 1], [0.2, 0.9])})`;
              textColor = interpolate(correctSpring, [0, 1], [1, 1.2]) > 1.1 ? '#10b981' : activeTheme.textColor;
            } else {
              const alpha = interpolate(correctSpring, [0, 1], [0.05, 0.01]);
              cardBg = `rgba(15, 23, 42, ${alpha})`;
              borderStyle = `1.5px solid rgba(255, 255, 255, ${interpolate(correctSpring, [0, 1], [0.06, 0.02])})`;
              textColor = 'rgba(255,255,255,0.3)';
            }
          }

          return (
            <div
              key={`option-${idx}`}
              style={{
                background: cardBg,
                border: borderStyle,
                borderRadius: '16px',
                padding: '16px 20px',
                fontSize: '16px',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                transform: `scale(${scale})`,
                opacity,
                color: textColor,
                boxSizing: 'border-box',
                transition: 'border 0.4s ease, color 0.4s ease',
              }}
            >
              <div
                style={{
                  width: '28px',
                  height: '28px',
                  borderRadius: '50%',
                  background: frame >= revealFrameStart && isCorrect ? '#10b981' : 'rgba(255,255,255,0.1)',
                  color: frame >= revealFrameStart && isCorrect ? '#000000' : '#ffffff',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 800,
                  fontSize: '13px',
                  marginRight: '16px',
                  flexShrink: 0,
                  transition: 'background 0.4s ease',
                }}
              >
                {finalOptions.length === 1 ? '✓' : String.fromCharCode(65 + idx)}
              </div>
              <div style={{ flex: 1 }}>{opt}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
