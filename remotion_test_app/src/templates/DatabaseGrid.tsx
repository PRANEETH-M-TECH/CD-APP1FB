import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, spring } from 'remotion';
import { getTheme } from '../themeHelper';

interface DatabaseGridProps {
  table_title: string;
  headers: string[];
  rows: string[][];
  highlight_row_idx?: number;
  highlight_col_idx?: number;
  theme: string;
}

export const DatabaseGrid: React.FC<DatabaseGridProps> = ({
  table_title,
  headers = [],
  rows = [],
  highlight_row_idx = -1,
  highlight_col_idx = -1,
  theme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const activeTheme = getTheme(theme);

  // Title fade in
  const titleOpacity = interpolate(frame, [0, 15], [0, 1], { extrapolateRight: 'clamp' });
  const titleTranslateY = interpolate(frame, [0, 15], [-15, 0], { extrapolateRight: 'clamp' });

  // Grid container scale-up spring
  const gridSpring = spring({
    frame,
    fps,
    config: { stiffness: 90, damping: 15 }
  });
  const gridScale = interpolate(gridSpring, [0, 1], [0.93, 1]);
  const gridOpacity = interpolate(frame, [5, 20], [0, 1], { extrapolateRight: 'clamp' });

  // Staggered rows fade-in
  const rowSprings = rows.map((_, idx) => {
    return spring({
      frame: frame - (18 + idx * 8),
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
        justifyContent: 'center',
        padding: '50px 60px',
        boxSizing: 'border-box',
        fontFamily: activeTheme.fontFamily,
        color: activeTheme.textColor,
      }}
    >
      {/* Table Title */}
      <h2
        style={{
          fontSize: '32px',
          fontWeight: 800,
          opacity: titleOpacity,
          transform: `translateY(${titleTranslateY}px)`,
          margin: '0 0 30px 0',
          color: activeTheme.accentColor,
          textShadow: '0 4px 12px rgba(0,0,0,0.3)',
          letterSpacing: '-0.5px',
        }}
      >
        {table_title}
      </h2>

      {/* Grid Container */}
      <div
        style={{
          width: '90%',
          maxHeight: '380px',
          background: 'rgba(15, 23, 42, 0.45)',
          border: '2px solid rgba(255,255,255,0.06)',
          borderRadius: '24px',
          boxShadow: '0 16px 40px rgba(0,0,0,0.3)',
          transform: `scale(${gridScale})`,
          opacity: gridOpacity,
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* Table Header Row */}
        <div
          style={{
            display: 'flex',
            background: 'rgba(15, 23, 42, 0.75)',
            borderBottom: activeTheme.cardBorder,
            padding: '16px 20px',
          }}
        >
          {headers.map((h, colIdx) => (
            <div
              key={`header-${colIdx}`}
              style={{
                flex: 1,
                fontWeight: 800,
                fontSize: '16px',
                color: activeTheme.accentColor,
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
                textAlign: 'left',
              }}
            >
              {h}
            </div>
          ))}
        </div>

        {/* Table Rows Body */}
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {rows.map((row, rowIdx) => {
            const scale = rowSprings[rowIdx];
            const opacity = rowSprings[rowIdx];
            const isRowHighlighted = highlight_row_idx === rowIdx;

            return (
              <div
                key={`row-${rowIdx}`}
                style={{
                  display: 'flex',
                  padding: '16px 20px',
                  borderBottom: rowIdx < rows.length - 1 ? '1px solid rgba(255,255,255,0.05)' : 'none',
                  background: isRowHighlighted
                    ? `rgba(${activeTheme.accentColorRgb}, 0.12)`
                    : 'transparent',
                  opacity,
                  transform: `scale(${interpolate(scale, [0, 1], [0.98, 1])})`,
                  transition: 'background 0.3s ease',
                  alignItems: 'center',
                }}
              >
                {row.map((cell, colIdx) => {
                  const isCellHighlighted = isRowHighlighted || highlight_col_idx === colIdx;
                  return (
                    <div
                      key={`cell-${rowIdx}-${colIdx}`}
                      style={{
                        flex: 1,
                        fontSize: '15px',
                        fontWeight: isCellHighlighted ? 700 : 500,
                        color: isCellHighlighted ? activeTheme.accentColor : activeTheme.textColor,
                        textAlign: 'left',
                      }}
                    >
                      {cell}
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
