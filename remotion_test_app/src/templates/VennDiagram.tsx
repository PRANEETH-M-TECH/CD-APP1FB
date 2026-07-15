import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, spring } from 'remotion';
import { getTheme } from '../themeHelper';

interface VennDiagramProps {
  left: string[];
  right: string[];
  intersection: string[];
  left_title?: string;
  right_title?: string;
  theme: string;
}

export const VennDiagram: React.FC<VennDiagramProps> = ({
  left = [],
  right = [],
  intersection = [],
  left_title = 'A',
  right_title = 'B',
  theme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const activeTheme = getTheme(theme);

  // Circle sliding together animation
  const leftCircleTranslateX = interpolate(
    spring({
      frame,
      fps,
      config: { stiffness: 90, damping: 15 }
    }),
    [0, 1],
    [-100, -70] // overlap position
  );

  const rightCircleTranslateX = interpolate(
    spring({
      frame,
      fps,
      config: { stiffness: 90, damping: 15 }
    }),
    [0, 1],
    [100, 70] // overlap position
  );

  const circlesOpacity = interpolate(frame, [0, 15], [0, 0.45], { extrapolateRight: 'clamp' });

  // Staggered contents reveal
  const leftItemSprings = left.map((_, idx) => {
    return spring({
      frame: frame - (15 + idx * 8),
      fps,
      config: { stiffness: 120, damping: 14 }
    });
  });

  const rightItemSprings = right.map((_, idx) => {
    return spring({
      frame: frame - (25 + idx * 8),
      fps,
      config: { stiffness: 120, damping: 14 }
    });
  });

  const intersectItemSprings = intersection.map((_, idx) => {
    return spring({
      frame: frame - (40 + idx * 8),
      fps,
      config: { stiffness: 120, damping: 14 }
    });
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
      }}
    >
      {/* Title Headers */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          width: '600px',
          marginBottom: '20px',
          fontSize: '24px',
          fontWeight: 800,
        }}
      >
        <div style={{ color: activeTheme.accentColor }}>{left_title}</div>
        <div style={{ color: '#ffffff' }}>Comparison</div>
        <div style={{ color: activeTheme.accentColor }}>{right_title}</div>
      </div>

      {/* Overlapping Diagram Circles container */}
      <div
        style={{
          width: '600px',
          height: '400px',
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {/* Left Circle (Visual Only) */}
        <div
          style={{
            position: 'absolute',
            width: '320px',
            height: '320px',
            borderRadius: '50%',
            background: activeTheme.cardBackground,
            border: `3px solid ${activeTheme.accentColor}`,
            boxShadow: `0 12px 32px rgba(${activeTheme.accentColorRgb}, 0.15)`,
            transform: `translateX(${leftCircleTranslateX}px)`,
            opacity: circlesOpacity,
            zIndex: 1,
          }}
        />

        {/* Right Circle (Visual Only) */}
        <div
          style={{
            position: 'absolute',
            width: '320px',
            height: '320px',
            borderRadius: '50%',
            background: activeTheme.cardBackground,
            border: `3px solid ${activeTheme.accentColor}`,
            boxShadow: `0 12px 32px rgba(${activeTheme.accentColorRgb}, 0.15)`,
            transform: `translateX(${rightCircleTranslateX}px)`,
            opacity: circlesOpacity,
            zIndex: 1,
          }}
        />

        {/* --- Content Overlay Columns --- */}
        
        {/* Left Circle Content */}
        <div
          style={{
            position: 'absolute',
            width: '180px',
            left: '30px',
            top: '80px',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
            zIndex: 10,
          }}
        >
          {left.map((item, idx) => {
            const scale = leftItemSprings[idx];
            return (
              <div
                key={`left-${idx}`}
                style={{
                  fontSize: '14px',
                  fontWeight: 600,
                  background: 'rgba(15, 23, 42, 0.75)',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                  borderRadius: '10px',
                  padding: '10px 14px',
                  transform: `scale(${scale})`,
                  opacity: scale,
                  boxShadow: '0 4px 8px rgba(0,0,0,0.15)',
                }}
              >
                {item}
              </div>
            );
          })}
        </div>

        {/* Intersection Content (Middle Overlap) */}
        <div
          style={{
            position: 'absolute',
            width: '180px',
            top: '80px',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
            alignItems: 'center',
            zIndex: 10,
          }}
        >
          {intersection.map((item, idx) => {
            const scale = intersectItemSprings[idx];
            return (
              <div
                key={`intersect-${idx}`}
                style={{
                  fontSize: '14px',
                  fontWeight: 700,
                  background: 'rgba(15, 23, 42, 0.9)',
                  border: `1.5px dashed ${activeTheme.accentColor}`,
                  borderRadius: '10px',
                  padding: '10px 14px',
                  transform: `scale(${scale})`,
                  opacity: scale,
                  boxShadow: `0 6px 16px rgba(${activeTheme.accentColorRgb}, 0.25)`,
                  textAlign: 'center',
                }}
              >
                {item}
              </div>
            );
          })}
        </div>

        {/* Right Circle Content */}
        <div
          style={{
            position: 'absolute',
            width: '180px',
            right: '30px',
            top: '80px',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
            zIndex: 10,
          }}
        >
          {right.map((item, idx) => {
            const scale = rightItemSprings[idx];
            return (
              <div
                key={`right-${idx}`}
                style={{
                  fontSize: '14px',
                  fontWeight: 600,
                  background: 'rgba(15, 23, 42, 0.75)',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                  borderRadius: '10px',
                  padding: '10px 14px',
                  transform: `scale(${scale})`,
                  opacity: scale,
                  boxShadow: '0 4px 8px rgba(0,0,0,0.15)',
                }}
              >
                {item}
              </div>
            );
          })}
        </div>

      </div>
    </div>
  );
};
