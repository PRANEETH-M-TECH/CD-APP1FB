import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, spring } from 'remotion';
import { getTheme } from '../themeHelper';

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

interface TitleSlideProps {
  title: string;
  subtitle: string;
  icon_name?: string;
  theme: string;
}

export const TitleSlide: React.FC<TitleSlideProps> = ({
  title,
  subtitle,
  icon_name = 'book-open',
  theme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const activeTheme = getTheme(theme);

  // 1. spring animation for icon scale (pop & overshoot)
  const iconScale = spring({
    frame,
    fps,
    config: {
      stiffness: activeTheme.stiffness,
      damping: activeTheme.damping,
      mass: activeTheme.mass
    }
  });

  // 2. spring animation for title slide up (elastic settle)
  const titleTranslateY = interpolate(
    spring({
      frame: frame - 8,
      fps,
      config: { stiffness: 100, damping: 14 }
    }),
    [0, 1],
    [50, 0]
  );

  const titleOpacity = interpolate(frame, [8, 18], [0, 1], { extrapolateRight: 'clamp' });

  // 3. subtitle fade in
  const subtitleOpacity = interpolate(frame, [18, 30], [0, 1], { extrapolateRight: 'clamp' });
  const subtitleTranslateY = interpolate(frame, [18, 30], [15, 0], { extrapolateRight: 'clamp' });

  // 4. slow idle animations
  const floatY = Math.sin((frame / 45) * Math.PI) * 6;
  const slowRotation = (frame * 0.1) % 360;

  const svgContent = ICON_SVGS[icon_name.toLowerCase()] || ICON_SVGS['book-open'];

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '60px',
        boxSizing: 'border-box',
        fontFamily: activeTheme.fontFamily
      }}
    >
      {/* Icon Card with glow and spring scaling */}
      <div
        style={{
          width: '140px',
          height: '140px',
          background: `radial-gradient(circle at 30% 30%, rgba(${activeTheme.accentColorRgb}, 0.25) 0%, rgba(${activeTheme.accentColorRgb}, 0.05) 100%)`,
          border: activeTheme.cardBorder,
          borderRadius: '32px',
          boxShadow: `0 16px 40px rgba(${activeTheme.accentColorRgb}, 0.2), inset 0 0 16px rgba(${activeTheme.accentColorRgb}, 0.1)`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '30px',
          boxSizing: 'border-box',
          transform: `scale(${iconScale}) translateY(${floatY}px) rotate(${slowRotation}deg)`,
          marginBottom: '40px',
        }}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke={activeTheme.accentColor}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{
            width: '100%',
            height: '100%',
            filter: `drop-shadow(0 0 8px ${activeTheme.accentColor})`,
          }}
        >
          {svgContent}
        </svg>
      </div>

      {/* Title with spring slide-up */}
      <h1
        style={{
          fontSize: '56px',
          fontWeight: 900,
          textAlign: 'center',
          margin: '0 0 16px 0',
          color: activeTheme.textColor,
          textShadow: '0 4px 16px rgba(0, 0, 0, 0.4)',
          transform: `translateY(${titleTranslateY}px)`,
          opacity: titleOpacity,
          letterSpacing: '-1.5px',
        }}
      >
        {title}
      </h1>

      {/* Subtitle with soft transition */}
      <p
        style={{
          fontSize: '22px',
          fontWeight: 500,
          textAlign: 'center',
          margin: 0,
          color: 'rgba(255, 255, 255, 0.7)',
          transform: `translateY(${subtitleTranslateY}px)`,
          opacity: subtitleOpacity,
          textShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
          maxWidth: '850px',
          lineHeight: '1.4',
        }}
      >
        {subtitle}
      </p>
    </div>
  );
};
