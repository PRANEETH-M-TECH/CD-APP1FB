import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, spring } from 'remotion';
import { getTheme } from '../themeHelper';

interface CycleTemplateProps {
  title: string;
  stages: any[];
  theme: string;
}

export const CycleTemplate: React.FC<CycleTemplateProps> = ({
  title,
  stages = [],
  theme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const activeTheme = getTheme(theme);
  const N = stages.length;

  // Title fade in
  const titleOpacity = interpolate(frame, [0, 15], [0, 1], { extrapolateRight: 'clamp' });
  const titleTranslateY = interpolate(frame, [0, 15], [-20, 0], { extrapolateRight: 'clamp' });

  // Circle path drawing animation
  const circleProgress = spring({
    frame: frame - 15,
    fps,
    config: { stiffness: 70, damping: 14 }
  });

  // Staggered stage bubbles pop in
  const stageSprings = stages.map((_, idx) => {
    return spring({
      frame: frame - (25 + idx * 12),
      fps,
      config: { stiffness: 120, damping: 14 }
    });
  });

  // Rotating highlight orbit
  const highlightOffset = (frame * 2.5) % 360;

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
          fontSize: '36px',
          fontWeight: 800,
          textAlign: 'center',
          margin: '0 0 30px 0',
          opacity: titleOpacity,
          transform: `translateY(${titleTranslateY}px)`,
          color: activeTheme.accentColor,
        }}
      >
        {title}
      </h2>

      {/* Main Cycle Canvas */}
      <div
        style={{
          width: '400px',
          height: '400px',
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {/* Background Circle Line (drawing itself) */}
        <svg
          style={{
            position: 'absolute',
            width: '100%',
            height: '100%',
            top: 0,
            left: 0,
            transform: 'rotate(-90deg)',
          }}
        >
          <circle
            cx="200"
            cy="200"
            r="140"
            fill="none"
            stroke={activeTheme.accentColor}
            strokeWidth="4"
            strokeDasharray={2 * Math.PI * 140}
            strokeDashoffset={2 * Math.PI * 140 * (1 - circleProgress)}
            opacity={circleProgress > 0 ? 0.35 : 0}
          />
          
          {/* Flow Indicator Dot */}
          {circleProgress >= 1 && (
            <circle
              cx={200 + 140 * Math.cos((highlightOffset * Math.PI) / 180)}
              cy={200 + 140 * Math.sin((highlightOffset * Math.PI) / 180)}
              r="7"
              fill={activeTheme.accentColor}
              style={{
                filter: `drop-shadow(0 0 6px ${activeTheme.accentColor})`,
              }}
            />
          )}
        </svg>

        {/* Stages (Placed in a circle) */}
        {stages.map((stage, idx) => {
          // Angle calculation for N nodes
          const angle = (idx * (2 * Math.PI)) / N - Math.PI / 2;
          const r = 140; // radius matches SVG circle
          const x = r * Math.cos(angle);
          const y = r * Math.sin(angle);

          const scale = stageSprings[idx];
          const opacity = stageSprings[idx];

          return (
            <div
              key={`stage-${idx}`}
              style={{
                position: 'absolute',
                width: '120px',
                height: '120px',
                transform: `translate(${x}px, ${y}px) scale(${scale})`,
                opacity,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexDirection: 'column',
                boxSizing: 'border-box',
                background: activeTheme.cardBackground,
                border: activeTheme.cardBorder,
                boxShadow: `0 8px 24px rgba(0,0,0,0.3), inset 0 0 8px rgba(${activeTheme.accentColorRgb}, 0.1)`,
                borderRadius: '24px',
                padding: '12px',
                zIndex: 20,
              }}
            >
              {/* Step Sequence Bubble */}
              <div
                style={{
                  fontSize: '11px',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  color: activeTheme.accentColor,
                  marginBottom: '6px',
                }}
              >
                Step {idx + 1}
              </div>
              <div
                style={{
                  fontSize: '14px',
                  fontWeight: 700,
                  textAlign: 'center',
                  lineHeight: '1.2',
                }}
              >
                {typeof stage === 'object' && stage !== null ? (stage as any).label : stage}
              </div>
            </div>
          );
        })}

        {/* Center label (Flow indicator arrow icon) */}
        <div
          style={{
            position: 'absolute',
            fontSize: '12px',
            textTransform: 'uppercase',
            fontWeight: 800,
            letterSpacing: '1px',
            opacity: circleProgress,
            color: 'rgba(255, 255, 255, 0.4)',
          }}
        >
          🔄 Cycle Flow
        </div>
      </div>
    </div>
  );
};
