import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, spring } from 'remotion';
import { getTheme } from '../themeHelper';

interface BranchData {
  label: string;
  leaves: string[];
}

interface TaxonomyTreeProps {
  title: string;
  root_label: string;
  branches: BranchData[];
  theme: string;
}

export const TaxonomyTree: React.FC<TaxonomyTreeProps> = ({
  title,
  root_label,
  branches = [],
  theme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const activeTheme = getTheme(theme);
  const N_branches = branches.length;

  // Title fade in
  const titleOpacity = interpolate(frame, [0, 15], [0, 1], { extrapolateRight: 'clamp' });
  const titleTranslateY = interpolate(frame, [0, 15], [-15, 0], { extrapolateRight: 'clamp' });

  // Root Node Drop-down spring
  const rootScale = spring({
    frame,
    fps,
    config: { stiffness: 100, damping: 14 }
  });
  const rootTranslateY = interpolate(rootScale, [0, 1], [-40, 0]);

  // Branch Nodes slide out and scale
  const branchSprings = branches.map((_, idx) => {
    return spring({
      frame: frame - (15 + idx * 8),
      fps,
      config: { stiffness: 120, damping: 14 }
    });
  });

  // Connection Lines progress
  const linesProgress = spring({
    frame: frame - 12,
    fps,
    config: { stiffness: 90, damping: 15 }
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
      {/* Title */}
      <h2
        style={{
          fontSize: '34px',
          fontWeight: 800,
          margin: '0 0 35px 0',
          opacity: titleOpacity,
          transform: `translateY(${titleTranslateY}px)`,
          color: activeTheme.accentColor,
        }}
      >
        {title}
      </h2>

      {/* Hierarchical Tree Container */}
      <div
        style={{
          width: '750px',
          height: '420px',
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        {/* SVG branches connectors layer */}
        <svg
          style={{
            position: 'absolute',
            width: '100%',
            height: '100%',
            top: 0,
            left: 0,
            zIndex: 1,
            pointerEvents: 'none',
          }}
        >
          {branches.map((_, idx) => {
            const x1 = 375;
            const y1 = 60;
            const w = 700;
            const step = w / N_branches;
            const x2 = step / 2 + idx * step;
            const y2 = 180;

            const curX = interpolate(linesProgress, [0, 1], [x1, x2]);
            const curY = interpolate(linesProgress, [0, 1], [y1, y2]);

            return (
              <g key={`lines-group-${idx}`}>
                <line
                  x1={x1}
                  y1={y1}
                  x2={curX}
                  y2={curY}
                  stroke={activeTheme.accentColor}
                  strokeWidth="3"
                  opacity={linesProgress > 0 ? 0.45 : 0}
                />
                {/* Horizontal branch helper line */}
                {linesProgress >= 1 && (
                  <line
                    x1={x2}
                    y1={y2}
                    x2={x2}
                    y2={interpolate(frame - 30, [0, 15], [y2, y2 + 60], { extrapolateRight: 'clamp' })}
                    stroke={activeTheme.accentColor}
                    strokeWidth="1.5"
                    strokeDasharray="4,3"
                    opacity={0.4}
                  />
                )}
              </g>
            );
          })}
        </svg>

        {/* Level 1: Root Node */}
        <div
          style={{
            position: 'absolute',
            top: '10px',
            padding: '16px 36px',
            background: activeTheme.accentColor,
            color: '#000000',
            borderRadius: '20px',
            fontWeight: 800,
            fontSize: '20px',
            boxShadow: `0 8px 24px rgba(${activeTheme.accentColorRgb}, 0.35)`,
            transform: `scale(${rootScale}) translateY(${rootTranslateY}px)`,
            opacity: rootScale,
            zIndex: 10,
          }}
        >
          {root_label}
        </div>

        {/* Level 2 & 3: Branches and Leaves Columns */}
        <div
          style={{
            position: 'absolute',
            top: '150px',
            display: 'flex',
            width: '100%',
            justifyContent: 'space-between',
            boxSizing: 'border-box',
            zIndex: 10,
          }}
        >
          {branches.map((branch, idx) => {
            const scale = branchSprings[idx];
            const opacity = branchSprings[idx];
            const translateY = interpolate(scale, [0, 1], [25, 0]);

            return (
              <div
                key={`branch-${idx}`}
                style={{
                  width: `${100 / N_branches - 2}%`,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  transform: `scale(${scale}) translateY(${translateY}px)`,
                  opacity,
                }}
              >
                {/* Branch Node Header */}
                <div
                  style={{
                    background: 'rgba(15, 23, 42, 0.95)',
                    border: `2.5px solid ${activeTheme.accentColor}`,
                    borderRadius: '14px',
                    padding: '12px 18px',
                    fontWeight: 700,
                    fontSize: '16px',
                    textAlign: 'center',
                    marginBottom: '20px',
                    boxShadow: '0 6px 16px rgba(0,0,0,0.2)',
                  }}
                >
                  {branch.label}
                </div>

                {/* Level 3: Leaf Cards */}
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '8px',
                    width: '90%',
                  }}
                >
                  {branch.leaves.map((leaf, leafIdx) => {
                    const leafAnim = spring({
                      frame: frame - (35 + idx * 6 + leafIdx * 6),
                      fps,
                      config: { stiffness: 140, damping: 15 }
                    });

                    return (
                      <div
                        key={`leaf-${idx}-${leafIdx}`}
                        style={{
                          background: activeTheme.cardBackground,
                          border: activeTheme.cardBorder,
                          borderRadius: '10px',
                          padding: '8px 12px',
                          fontSize: '13px',
                          textAlign: 'center',
                          fontWeight: 500,
                          transform: `scale(${leafAnim})`,
                          opacity: leafAnim,
                          boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
                        }}
                      >
                        {leaf}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
