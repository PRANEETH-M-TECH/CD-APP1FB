import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, spring } from 'remotion';
import { getTheme } from '../themeHelper';

interface BeforeAfterSliderProps {
  title: string;
  before_label?: string;
  after_label?: string;
  before_text?: string;
  after_text?: string;
  theme: string;
}

export const BeforeAfterSlider: React.FC<BeforeAfterSliderProps> = ({
  title,
  before_label = 'Before',
  after_label = 'After',
  before_text = 'Initial State',
  after_text = 'Resulting State',
  theme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const activeTheme = getTheme(theme);

  // Title fade-in
  const titleOpacity = interpolate(frame, [0, 15], [0, 1], { extrapolateRight: 'clamp' });
  const titleTranslateY = interpolate(frame, [0, 15], [-15, 0], { extrapolateRight: 'clamp' });

  // Slider wipe progress (moves from left to right)
  const sliderProgress = spring({
    frame: frame - 15,
    fps,
    config: { stiffness: 60, damping: 15 } // smooth wipe
  });

  const dividerX = interpolate(sliderProgress, [0, 1], [0, 100]); // percentage position

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
          opacity: titleOpacity,
          transform: `translateY(${titleTranslateY}px)`,
          margin: '0 0 35px 0',
          color: activeTheme.accentColor,
        }}
      >
        {title}
      </h2>

      {/* Before / After Wipe Slider Canvas */}
      <div
        style={{
          width: '720px',
          height: '340px',
          position: 'relative',
          borderRadius: '24px',
          border: '2px solid rgba(255,255,255,0.06)',
          boxShadow: '0 16px 40px rgba(0,0,0,0.3)',
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {/* LEFT COLUMN: Before State (Base background) */}
        <div
          style={{
            position: 'absolute',
            width: '100%',
            height: '100%',
            top: 0,
            left: 0,
            background: 'linear-gradient(135deg, #1e1e24 0%, #111115 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1,
          }}
        >
          <div style={{ textAlign: 'center', maxWidth: '300px', transform: 'translateX(-150px)' }}>
            <h4
              style={{
                fontSize: '22px',
                fontWeight: 800,
                color: '#f43f5e',
                margin: '0 0 8px 0',
                textTransform: 'uppercase',
              }}
            >
              {before_label}
            </h4>
            <p style={{ fontSize: '15px', color: 'rgba(255,255,255,0.6)', margin: 0 }}>
              {before_text}
            </p>
          </div>
        </div>

        {/* RIGHT COLUMN: After State (Overlay with clip-path) */}
        <div
          style={{
            position: 'absolute',
            width: '100%',
            height: '100%',
            top: 0,
            left: 0,
            background: `linear-gradient(135deg, rgba(${activeTheme.accentColorRgb}, 0.2) 0%, rgba(15, 23, 42, 0.95) 100%)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 2,
            clipPath: `polygon(${dividerX}% 0%, 100% 0%, 100% 100%, ${dividerX}% 100%)`,
          }}
        >
          <div style={{ textAlign: 'center', maxWidth: '300px', transform: 'translateX(150px)' }}>
            <h4
              style={{
                fontSize: '22px',
                fontWeight: 800,
                color: activeTheme.accentColor,
                margin: '0 0 8px 0',
                textTransform: 'uppercase',
              }}
            >
              {after_label}
            </h4>
            <p style={{ fontSize: '15px', color: 'rgba(255,255,255,0.8)', margin: 0 }}>
              {after_text}
            </p>
          </div>
        </div>

        {/* Sliding vertical divider bar */}
        <div
          style={{
            position: 'absolute',
            left: `${dividerX}%`,
            top: 0,
            width: '4px',
            height: '100%',
            background: activeTheme.accentColor,
            boxShadow: `0 0 16px ${activeTheme.accentColor}`,
            zIndex: 5,
            opacity: sliderProgress > 0 ? 1 : 0,
          }}
        />
      </div>
    </div>
  );
};
