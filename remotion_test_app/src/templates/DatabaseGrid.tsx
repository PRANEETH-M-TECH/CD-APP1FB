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

interface DatabaseGridProps {
  table_title: string;
  headers: string[];
  rows: string[][];
  highlight_row_idx?: number;
  highlight_col_idx?: number;
  theme: 'indigo' | 'gold' | 'emerald' | 'rose';
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
  const { durationInFrames } = useVideoConfig();

  const accentColor = THEME_ACCENTS[theme] || THEME_ACCENTS.indigo;
  const accentRgb = THEME_ACCENT_RGBS[theme] || THEME_ACCENT_RGBS.indigo;

  // --- Animation Timing Configuration ---
  const headerFadeStart = 0;
  const headerFadeEnd = 15;
  const titleOpacity = interpolate(frame, [headerFadeStart, headerFadeEnd], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Table container fades in and scales slightly
  const tableStart = 10;
  const tableEnd = 25;
  const tableOpacity = interpolate(frame, [tableStart, tableEnd], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const tableScale = interpolate(frame, [tableStart, tableEnd], [0.95, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  // Rows fade in one-by-one
  const rowOpacities = rows.map((_, idx) => {
    const start = tableEnd + idx * 8;
    const end = start + 10;
    return interpolate(frame, [start, end], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
  });

  // Highlight pulse animation (glow gets stronger and weaker)
  const pulseScale = 1 + Math.sin((frame / 10) * Math.PI) * 0.03;

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
          fontSize: '32px',
          fontWeight: 800,
          color: '#ffffff',
          textAlign: 'center',
          opacity: titleOpacity,
          margin: '0 0 35px 0',
          textTransform: 'uppercase',
          letterSpacing: '1px',
          textShadow: '0 4px 8px rgba(0,0,0,0.5)',
        }}
      >
        {table_title}
      </h2>

      {/* Grid Container */}
      <div
        style={{
          width: '85%',
          opacity: tableOpacity,
          transform: `scale(${tableScale})`,
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1.5px solid rgba(255, 255, 255, 0.06)',
          borderRadius: '16px',
          overflow: 'hidden',
          boxShadow: '0 16px 48px rgba(0,0,0,0.4)',
        }}
      >
        <table
          style={{
            width: '100%',
            borderCollapse: 'collapse',
            textAlign: 'left',
            fontFamily: 'system-ui, sans-serif',
          }}
        >
          {/* Table Headers */}
          <thead>
            <tr
              style={{
                borderBottom: `2.5px solid ${accentColor}`,
                background: `rgba(${accentRgb}, 0.08)`,
              }}
            >
              {headers.map((head, idx) => (
                <th
                  key={`th-${idx}`}
                  style={{
                    padding: '16px 24px',
                    fontSize: '18px',
                    fontWeight: 800,
                    color: accentColor,
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                  }}
                >
                  {head}
                </th>
              ))}
            </tr>
          </thead>

          {/* Table Rows */}
          <tbody>
            {rows.map((row, rowIdx) => {
              const rowOpacity = rowOpacities[rowIdx] ?? 0;

              return (
                <tr
                  key={`tr-${rowIdx}`}
                  style={{
                    borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                    opacity: rowOpacity,
                    transform: `translateY(${interpolate(frame, [tableEnd + rowIdx * 8, tableEnd + rowIdx * 8 + 10], [10, 0], { extrapolateRight: 'clamp' })}px)`,
                    background: rowIdx % 2 === 0 ? 'transparent' : 'rgba(255, 255, 255, 0.01)',
                  }}
                >
                  {row.map((cell, colIdx) => {
                    const isHighlighted = rowIdx === highlight_row_idx && colIdx === highlight_col_idx;

                    return (
                      <td
                        key={`td-${rowIdx}-${colIdx}`}
                        style={{
                          padding: '16px 24px',
                          fontSize: '16px',
                          fontWeight: 500,
                          color: cell === 'NULL' ? '#ef4444' : '#e2e8f0',
                          position: 'relative',
                          zIndex: isHighlighted ? 20 : 1,
                        }}
                      >
                        {isHighlighted ? (
                          /* Highlight wrapper with pulse effect */
                          <div
                            style={{
                              position: 'absolute',
                              top: '4px',
                              left: '4px',
                              right: '4px',
                              bottom: '4px',
                              border: `2px solid ${accentColor}`,
                              background: `rgba(${accentRgb}, 0.15)`,
                              borderRadius: '6px',
                              zIndex: -1,
                              transform: `scale(${pulseScale})`,
                              boxShadow: `0 0 16px rgba(${accentRgb}, 0.4)`,
                              pointerEvents: 'none',
                              display: 'flex',
                              alignItems: 'center',
                              padding: '12px 20px',
                            }}
                          />
                        ) : null}
                        {cell}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
