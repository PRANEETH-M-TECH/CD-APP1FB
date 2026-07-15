import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, spring } from 'remotion';
import { getTheme } from '../themeHelper';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface MathDerivationProps {
  title: string;
  formula: string;
  steps: string[];
  theme: string;
}

export const MathDerivation: React.FC<MathDerivationProps> = ({
  title,
  formula,
  steps = [],
  theme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const activeTheme = getTheme(theme);

  // Title fade in
  const titleOpacity = interpolate(frame, [0, 15], [0, 1], { extrapolateRight: 'clamp' });
  const titleTranslateY = interpolate(frame, [0, 15], [-15, 0], { extrapolateRight: 'clamp' });

  // Main formula spring scale
  const mainFormulaScale = spring({
    frame: frame - 12,
    fps,
    config: { stiffness: 100, damping: 14 }
  });
  const mainFormulaOpacity = interpolate(frame, [12, 22], [0, 1], { extrapolateRight: 'clamp' });

  // Steps sequential pop-in
  const stepSprings = steps.map((_, idx) => {
    return spring({
      frame: frame - (24 + idx * 16),
      fps,
      config: { stiffness: 120, damping: 14 }
    });
  });

  // Render LaTeX to HTML string safely
  const renderLaTeX = (latexStr: string) => {
    try {
      const clean = latexStr.trim().replace(/^\$+|\$+$/g, '');
      return katex.renderToString(clean, { throwOnError: false });
    } catch (e) {
      return latexStr; // Fallback to raw text
    }
  };

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
      }}
    >
      {/* Title */}
      <h2
        style={{
          fontSize: '34px',
          fontWeight: 800,
          margin: '0 0 24px 0',
          opacity: titleOpacity,
          transform: `translateY(${titleTranslateY}px)`,
          color: activeTheme.accentColor,
        }}
      >
        {title}
      </h2>

      {/* Main Formula Board */}
      {formula && (
        <div
          style={{
            background: 'rgba(15, 23, 42, 0.6)',
            border: activeTheme.cardBorder,
            borderRadius: '20px',
            padding: '24px 40px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '32px',
            marginBottom: '35px',
            boxShadow: `0 12px 36px rgba(0,0,0,0.3), inset 0 0 12px rgba(${activeTheme.accentColorRgb}, 0.05)`,
            transform: `scale(${mainFormulaScale})`,
            opacity: mainFormulaOpacity,
          }}
          dangerouslySetInnerHTML={{ __html: renderLaTeX(formula) }}
        />
      )}

      {/* Solving Steps list */}
      <div
        style={{
          width: '85%',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px',
        }}
      >
        {steps.map((step, idx) => {
          const scale = stepSprings[idx];
          const opacity = stepSprings[idx];
          const translateY = interpolate(scale, [0, 1], [15, 0]);

          return (
            <div
              key={`step-${idx}`}
              style={{
                display: 'flex',
                alignItems: 'center',
                background: activeTheme.cardBackground,
                border: activeTheme.cardBorder,
                borderRadius: '16px',
                padding: '16px 24px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                transform: `scale(${scale}) translateY(${translateY}px)`,
                opacity,
                boxSizing: 'border-box',
              }}
            >
              {/* Step counter */}
              <div
                style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  background: activeTheme.accentColor,
                  color: '#000000',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 800,
                  fontSize: '14px',
                  marginRight: '20px',
                  boxShadow: `0 4px 12px rgba(${activeTheme.accentColorRgb}, 0.3)`,
                  flexShrink: 0,
                }}
              >
                {idx + 1}
              </div>

              {/* Step equation/derivation string */}
              <div
                style={{
                  fontSize: '20px',
                  fontWeight: 600,
                  width: '100%',
                }}
                dangerouslySetInnerHTML={{ __html: renderLaTeX(step) }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
};
