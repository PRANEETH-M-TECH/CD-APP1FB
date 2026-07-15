import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, spring } from 'remotion';
import { getTheme } from '../themeHelper';

interface ColumnData {
  header: string;
  bullets: string[];
}

interface ColumnComparisonProps {
  left_column: ColumnData;
  right_column: ColumnData;
  theme: string;
}

export const ColumnComparison: React.FC<ColumnComparisonProps> = ({
  left_column,
  right_column,
  theme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const activeTheme = getTheme(theme);

  // Left column slide-in spring
  const leftSpring = spring({
    frame,
    fps,
    config: {
      stiffness: activeTheme.stiffness,
      damping: activeTheme.damping,
      mass: activeTheme.mass
    }
  });
  const leftX = interpolate(leftSpring, [0, 1], [-150, 0]);
  const leftOpacity = leftSpring;

  // Right column slide-in spring (staggered slightly)
  const rightSpring = spring({
    frame: frame - 6,
    fps,
    config: {
      stiffness: activeTheme.stiffness,
      damping: activeTheme.damping,
      mass: activeTheme.mass
    }
  });
  const rightX = interpolate(rightSpring, [0, 1], [150, 0]);
  const rightOpacity = rightSpring;

  // Divider expansion
  const dividerHeight = interpolate(
    spring({
      frame: frame - 12,
      fps,
      config: { stiffness: 90, damping: 15 }
    }),
    [0, 1],
    [0, 80] // height percentage
  );

  // Left side staggered bullets
  const leftBulletSprings = (left_column.bullets || []).map((_, idx) => {
    return spring({
      frame: frame - (15 + idx * 8),
      fps,
      config: { stiffness: 120, damping: 14 }
    });
  });

  // Right side staggered bullets
  const rightBulletSprings = (right_column.bullets || []).map((_, idx) => {
    return spring({
      frame: frame - (21 + idx * 8),
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
        alignItems: 'center',
        justifyContent: 'center',
        padding: '50px 60px',
        boxSizing: 'border-box',
        fontFamily: activeTheme.fontFamily,
        color: activeTheme.textColor,
      }}
    >
      <div
        style={{
          width: '100%',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'stretch',
          position: 'relative',
        }}
      >
        {/* Left Column Card */}
        <div
          style={{
            width: '46%',
            background: activeTheme.cardBackground,
            border: activeTheme.cardBorder,
            borderRadius: '24px',
            padding: '30px 24px',
            boxSizing: 'border-box',
            boxShadow: '0 12px 32px rgba(0, 0, 0, 0.25)',
            transform: `translateX(${leftX}px)`,
            opacity: leftOpacity,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <h3
            style={{
              fontSize: '26px',
              fontWeight: 800,
              margin: '0 0 20px 0',
              color: activeTheme.accentColor,
              borderBottom: `2px solid rgba(${activeTheme.accentColorRgb}, 0.2)`,
              paddingBottom: '12px',
            }}
          >
            {left_column.header}
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {(left_column.bullets || []).map((bullet, idx) => {
              const scale = leftBulletSprings[idx];
              return (
                <div
                  key={`l-bullet-${idx}`}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    fontSize: '16px',
                    lineHeight: '1.4',
                    fontWeight: 500,
                    opacity: scale,
                    transform: `scale(${scale})`,
                    transformOrigin: 'left center',
                  }}
                >
                  <div
                    style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      background: activeTheme.accentColor,
                      marginTop: '7px',
                      marginRight: '12px',
                      flexShrink: 0,
                    }}
                  />
                  <span>{bullet}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Vertical Divider line */}
        <div
          style={{
            position: 'absolute',
            left: '50%',
            top: `calc(50% - ${dividerHeight / 2}%)`,
            width: '2px',
            height: `${dividerHeight}%`,
            background: `linear-gradient(to bottom, rgba(${activeTheme.accentColorRgb}, 0), ${activeTheme.accentColor}, rgba(${activeTheme.accentColorRgb}, 0))`,
            opacity: dividerHeight > 0 ? 0.5 : 0,
          }}
        />

        {/* Right Column Card */}
        <div
          style={{
            width: '46%',
            background: activeTheme.cardBackground,
            border: activeTheme.cardBorder,
            borderRadius: '24px',
            padding: '30px 24px',
            boxSizing: 'border-box',
            boxShadow: '0 12px 32px rgba(0, 0, 0, 0.25)',
            transform: `translateX(${rightX}px)`,
            opacity: rightOpacity,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <h3
            style={{
              fontSize: '26px',
              fontWeight: 800,
              margin: '0 0 20px 0',
              color: activeTheme.accentColor,
              borderBottom: `2px solid rgba(${activeTheme.accentColorRgb}, 0.2)`,
              paddingBottom: '12px',
            }}
          >
            {right_column.header}
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {(right_column.bullets || []).map((bullet, idx) => {
              const scale = rightBulletSprings[idx];
              return (
                <div
                  key={`r-bullet-${idx}`}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    fontSize: '16px',
                    lineHeight: '1.4',
                    fontWeight: 500,
                    opacity: scale,
                    transform: `scale(${scale})`,
                    transformOrigin: 'left center',
                  }}
                >
                  <div
                    style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      background: activeTheme.accentColor,
                      marginTop: '7px',
                      marginRight: '12px',
                      flexShrink: 0,
                    }}
                  />
                  <span>{bullet}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
