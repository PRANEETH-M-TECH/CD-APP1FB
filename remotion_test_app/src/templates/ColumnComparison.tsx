import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, Easing } from 'remotion';

const THEME_ACCENTS = {
  indigo: '#6366f1',
  gold: '#fbbf24',
  emerald: '#10b981',
  rose: '#f43f5e',
};

const THEME_ACCENT_RGBS = {
  indigo: '99, 102, 241',
  gold: '251, 191, 36',
  emerald: '16, 185, 129',
  rose: '244, 63, 94',
};

interface ColumnData {
  header: string;
  bullets: string[];
}

interface ColumnComparisonProps {
  left_column: ColumnData;
  right_column: ColumnData;
  theme: 'indigo' | 'gold' | 'emerald' | 'rose';
}

export const ColumnComparison: React.FC<ColumnComparisonProps> = ({
  left_column,
  right_column,
  theme,
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  const accentColor = THEME_ACCENTS[theme] || THEME_ACCENTS.indigo;
  const accentRgb = THEME_ACCENT_RGBS[theme] || THEME_ACCENT_RGBS.indigo;

  // --- Animation Timing Configuration ---
  const leftSlideStart = 0;
  const leftSlideEnd = 20;

  const rightSlideStart = 10;
  const rightSlideEnd = 30;

  // Card slide-ins
  const leftX = interpolate(frame, [leftSlideStart, leftSlideEnd], [-120, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  const leftOpacity = interpolate(frame, [leftSlideStart, leftSlideEnd], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const rightX = interpolate(frame, [rightSlideStart, rightSlideEnd], [120, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  const rightOpacity = interpolate(frame, [rightSlideStart, rightSlideEnd], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Vertical separator growth
  const dividerStart = 15;
  const dividerEnd = 35;
  const dividerProgress = interpolate(frame, [dividerStart, dividerEnd], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Staggered bullet points reveal
  const bulletsLeftOpacity = (left_column.bullets || []).map((_, idx) => {
    const start = leftSlideEnd + 5 + idx * 12;
    const end = start + 10;
    return interpolate(frame, [start, end], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
  });

  const bulletsRightOpacity = (right_column.bullets || []).map((_, idx) => {
    const start = rightSlideEnd + 5 + idx * 12;
    const end = start + 10;
    return interpolate(frame, [start, end], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
  });

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        padding: '50px 60px',
        boxSizing: 'border-box',
        alignItems: 'center',
        position: 'relative',
      }}
    >
      {/* Left Column Card */}
      <div
        style={{
          width: '46%',
          height: '80%',
          opacity: leftOpacity,
          transform: `translateX(${leftX}px)`,
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          borderRadius: '16px',
          padding: '30px 40px',
          boxSizing: 'border-box',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
        }}
      >
        <h3
          style={{
            fontSize: '28px',
            fontWeight: 800,
            color: accentColor,
            margin: '0 0 24px 0',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            borderBottom: `2px solid rgba(${accentRgb}, 0.25)`,
            paddingBottom: '12px',
          }}
        >
          {left_column.header}
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {(left_column.bullets || []).map((bullet, idx) => (
            <div
              key={`left-bullet-${idx}`}
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: '12px',
                opacity: bulletsLeftOpacity[idx] ?? 0,
              }}
            >
              <span style={{ color: accentColor, fontSize: '18px', lineHeight: '1.2' }}>•</span>
              <p style={{ fontSize: '16px', lineHeight: '1.5', color: '#e2e8f0', margin: 0, fontWeight: 500 }}>
                {bullet}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Center Divider Line */}
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: '50%',
          transform: 'translate(-50%, -50%)',
          width: '2px',
          height: `${70 * dividerProgress}%`,
          borderLeft: `2.5px dashed rgba(${accentRgb}, 0.3)`,
          pointerEvents: 'none',
        }}
      />

      {/* Right Column Card */}
      <div
        style={{
          width: '46%',
          marginLeft: '8%', // Shift past left column + spacing
          height: '80%',
          opacity: rightOpacity,
          transform: `translateX(${rightX}px)`,
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          borderRadius: '16px',
          padding: '30px 40px',
          boxSizing: 'border-box',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
        }}
      >
        <h3
          style={{
            fontSize: '28px',
            fontWeight: 800,
            color: accentColor,
            margin: '0 0 24px 0',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            borderBottom: `2px solid rgba(${accentRgb}, 0.25)`,
            paddingBottom: '12px',
          }}
        >
          {right_column.header}
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {(right_column.bullets || []).map((bullet, idx) => (
            <div
              key={`right-bullet-${idx}`}
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: '12px',
                opacity: bulletsRightOpacity[idx] ?? 0,
              }}
            >
              <span style={{ color: accentColor, fontSize: '18px', lineHeight: '1.2' }}>•</span>
              <p style={{ fontSize: '16px', lineHeight: '1.5', color: '#e2e8f0', margin: 0, fontWeight: 500 }}>
                {bullet}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
