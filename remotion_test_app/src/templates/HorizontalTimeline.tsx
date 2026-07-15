import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, Easing } from 'remotion';

// Simple Lucide SVGs matching the app's available icon list
const ICON_SVGS: Record<string, React.ReactNode> = {
  crown: <path d="M2 4l3 12h14l3-12-6 7-4-7-4 7-6-7zm3 16h14a1 1 0 0 1 1 1v1H4v-1a1 1 0 0 1 1-1z" />,
  shield: <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />,
  bell: <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9M13.73 21a2 2 0 0 1-3.46 0" />,
  info: (
    <>
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </>
  ),
  book: (
    <>
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
    </>
  ),
  'book-open': (
    <>
      <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
      <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
    </>
  ),
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </>
  ),
  database: (
    <>
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
      <path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3" />
    </>
  ),
  globe: (
    <>
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </>
  )
};

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

interface Stage {
  step_no: number;
  label: string;
  icon_name: string;
}

interface HorizontalTimelineProps {
  timeline_title: string;
  stages: Stage[];
  theme: 'indigo' | 'gold' | 'emerald' | 'rose';
}

export const HorizontalTimeline: React.FC<HorizontalTimelineProps> = ({
  timeline_title,
  stages = [],
  theme,
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  const accentColor = THEME_ACCENTS[theme] || THEME_ACCENTS.indigo;
  const accentRgb = THEME_ACCENT_RGBS[theme] || THEME_ACCENT_RGBS.indigo;

  const N = stages.length;

  // --- Animation Timing Configuration ---
  const headerFadeStart = 0;
  const headerFadeEnd = 15;
  const titleOpacity = interpolate(frame, [headerFadeStart, headerFadeEnd], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Timeline line draws from left (15%) to right (85%)
  const lineDrawStart = 15;
  const lineDrawEnd = 45;
  const lineProgress = interpolate(frame, [lineDrawStart, lineDrawEnd], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Stages scale up staggered after timeline reaches them
  const stageScaleValues = stages.map((_, idx) => {
    // Stagger based on position
    const startTrigger = lineDrawStart + (idx / (N - 1 || 1)) * (lineDrawEnd - lineDrawStart);
    const start = startTrigger + 2;
    const end = start + 12;

    return interpolate(frame, [start, end], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
      easing: Easing.bezier(0.16, 1, 0.3, 1),
    });
  });

  // Floating bobbing effect for timeline elements
  const bobbing = Math.sin((frame / 35) * Math.PI) * 5;

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '50px 60px',
        boxSizing: 'border-box',
      }}
    >
      {/* Title */}
      <h2
        style={{
          fontSize: '36px',
          fontWeight: 800,
          color: '#ffffff',
          textAlign: 'center',
          opacity: titleOpacity,
          margin: '0 0 80px 0',
          textTransform: 'uppercase',
          letterSpacing: '1px',
          textShadow: '0 4px 8px rgba(0,0,0,0.5)',
        }}
      >
        {timeline_title}
      </h2>

      {/* Timeline Track Container */}
      <div
        style={{
          position: 'relative',
          width: '80%',
          height: '250px',
          display: 'flex',
          alignItems: 'center',
          boxSizing: 'border-box',
        }}
      >
        {/* Connection Dashed Line */}
        <svg
          style={{
            position: 'absolute',
            left: 0,
            top: '50%',
            transform: 'translateY(-50%)',
            width: '100%',
            height: '20px',
            pointerEvents: 'none',
            zIndex: 1,
          }}
        >
          {/* Base inactive trace */}
          <line
            x1="5%"
            y1="50%"
            x2="95%"
            y2="50%"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="4"
            strokeLinecap="round"
          />
          {/* Active growing path */}
          <line
            x1="5%"
            y1="50%"
            x2={`${5 + 90 * lineProgress}%`}
            y2="50%"
            stroke={accentColor}
            strokeWidth="4"
            strokeDasharray="8,8"
            strokeLinecap="round"
            style={{
              filter: `drop-shadow(0 0 4px ${accentColor})`,
            }}
          />
        </svg>

        {/* Stages */}
        {stages.map((stage, idx) => {
          const leftPercent = 5 + (idx / (N - 1 || 1)) * 90; // Distribute evenly
          const scale = stageScaleValues[idx] ?? 0;
          const svgContent = ICON_SVGS[stage.icon_name.toLowerCase()] || ICON_SVGS.info;

          // Bob every odd node in the opposite direction
          const offsetBob = idx % 2 === 0 ? bobbing : -bobbing;

          return (
            <div
              key={idx}
              style={{
                position: 'absolute',
                left: `${leftPercent}%`,
                top: '50%',
                transform: `translate(-50%, -50%) translateScale(0, ${offsetBob}px) scale(${scale})`,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                zIndex: 10,
                width: '120px',
              }}
            >
              {/* Stage Circle with Icon */}
              <div
                style={{
                  position: 'relative',
                  width: '80px',
                  height: '80px',
                  borderRadius: '50%',
                  background: 'linear-gradient(135deg, #090d16 0%, #151824 100%)',
                  border: `3px solid ${accentColor}`,
                  boxShadow: `0 8px 24px rgba(0,0,0,0.5), 0 0 12px rgba(${accentRgb}, 0.15)`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '18px',
                  boxSizing: 'border-box',
                }}
              >
                {/* Step Number Badge */}
                <div
                  style={{
                    position: 'absolute',
                    top: '-6px',
                    right: '-6px',
                    width: '26px',
                    height: '26px',
                    borderRadius: '50%',
                    background: accentColor,
                    color: '#000000',
                    fontSize: '12px',
                    fontWeight: 900,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: '2px solid #090d16',
                    boxShadow: '0 2px 6px rgba(0,0,0,0.3)',
                  }}
                >
                  {stage.step_no}
                </div>

                {/* SVG Icon */}
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke={accentColor}
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  style={{
                    width: '100%',
                    height: '100%',
                    filter: `drop-shadow(0 0 3px ${accentColor})`,
                  }}
                >
                  {svgContent}
                </svg>
              </div>

              {/* Caption Label */}
              <p
                style={{
                  marginTop: '16px',
                  fontSize: '14px',
                  fontWeight: 700,
                  color: '#e2e8f0',
                  textAlign: 'center',
                  lineHeight: '1.3',
                  margin: '16px 0 0 0',
                  textTransform: 'capitalize',
                  maxWidth: '120px',
                  textShadow: '0 2px 4px rgba(0,0,0,0.6)',
                }}
              >
                {stage.label}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
};
