import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, spring } from 'remotion';
import { getTheme } from '../themeHelper';

interface ConceptDiagramProps {
  left_title?: string;
  left_bullets?: string[];
  central_node: string;
  leaf_nodes: string[];
  theme: string;
}

export const ConceptDiagram: React.FC<ConceptDiagramProps> = ({
  left_title = '',
  left_bullets = [],
  central_node,
  leaf_nodes = [],
  theme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const activeTheme = getTheme(theme);

  const hasBullets = left_bullets && left_bullets.length > 0;
  const N_leaves = leaf_nodes.length;

  // Coordinate geometry based on layout mode
  const centerX = hasBullets ? 900 : 640;
  const centerY = 360;
  const radius = hasBullets ? 190 : 270;

  // Helper to calculate exact branch angles
  const getLeafAngle = (idx: number, total: number) => {
    if (hasBullets) {
      // Keep original distribution (spread over a 120-degree arc on the right)
      return -Math.PI / 3 + (idx * (2 * Math.PI / 3)) / Math.max(1, total - 1);
    } else {
      // Centered layout: split left and right sides symmetrically
      const leftNodes: number[] = [];
      const rightNodes: number[] = [];
      for (let i = 0; i < total; i++) {
        if (i % 2 === 0) {
          rightNodes.push(i);
        } else {
          leftNodes.push(i);
        }
      }

      if (idx % 2 === 0) {
        // Right side node (spread between -40 and 40 degrees)
        const k = rightNodes.indexOf(idx);
        const M = rightNodes.length;
        return -Math.PI / 4.5 + (k * (2 * Math.PI / 4.5)) / Math.max(1, M - 1);
      } else {
        // Left side node (spread between 140 and 220 degrees)
        const k = leftNodes.indexOf(idx);
        const L = leftNodes.length;
        return Math.PI - Math.PI / 4.5 + (k * (2 * Math.PI / 4.5)) / Math.max(1, L - 1);
      }
    }
  };

  // --- Animation Timing & Springs ---
  
  // Left Column title fade-in (if bullets exist)
  const titleOpacity = interpolate(frame, [0, 12], [0, 1], { extrapolateRight: 'clamp' });
  const titleTranslateX = interpolate(frame, [0, 12], [-20, 0], { extrapolateRight: 'clamp' });

  // Staggered list items
  const bulletSprings = left_bullets.map((_, idx) => {
    return spring({
      frame: frame - (10 + idx * 8),
      fps,
      config: { stiffness: 120, damping: 14 }
    });
  });

  // Central Node scale-up
  const centerScale = spring({
    frame: frame - 10,
    fps,
    config: {
      stiffness: activeTheme.stiffness,
      damping: activeTheme.damping,
      mass: activeTheme.mass
    }
  });

  // SVG lines draw outward
  const lineProgress = spring({
    frame: frame - 20,
    fps,
    config: { stiffness: 90, damping: 15 }
  });

  // Outer Leaf Nodes pop-in
  const leafSprings = leaf_nodes.map((_, idx) => {
    return spring({
      frame: frame - (35 + idx * 6),
      fps,
      config: { stiffness: 120, damping: 14 }
    });
  });

  // Idle floating effect
  const floatY = Math.sin((frame / 45) * Math.PI) * 4;

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        padding: '60px 80px',
        boxSizing: 'border-box',
        fontFamily: activeTheme.fontFamily,
        alignItems: 'center',
        justifyContent: 'space-between',
        color: activeTheme.textColor,
        position: 'relative',
      }}
    >
      {/* Left Column: Conceptual definitions (only shown if bullets are present) */}
      {hasBullets && (
        <div
          style={{
            width: '40%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            zIndex: 5,
          }}
        >
          <h2
            style={{
              fontSize: '38px',
              fontWeight: 800,
              margin: '0 0 24px 0',
              opacity: titleOpacity,
              transform: `translateX(${titleTranslateX}px)`,
              letterSpacing: '-1px',
              color: activeTheme.accentColor,
            }}
          >
            {left_title}
          </h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {left_bullets.map((bullet, idx) => {
              const opacity = bulletSprings[idx];
              const translateY = interpolate(bulletSprings[idx], [0, 1], [15, 0]);
              
              return (
                <div
                  key={`bullet-${idx}`}
                  style={{
                    padding: '16px 20px',
                    background: activeTheme.cardBackground,
                    border: activeTheme.cardBorder,
                    borderRadius: '16px',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                    fontSize: '18px',
                    lineHeight: '1.4',
                    fontWeight: 500,
                    opacity,
                    transform: `translateY(${translateY}px)`,
                  }}
                >
                  {bullet}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Mind Map Canvas (Takes full screen if no bullets, else takes right half) */}
      <div
        style={{
          position: hasBullets ? 'relative' : 'absolute',
          width: hasBullets ? '55%' : '100%',
          height: '100%',
          top: 0,
          left: hasBullets ? 'auto' : 0,
          right: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {/* SVG Drawing Canvas (1280x720 absolute workspace coordinates) */}
        <svg
          viewBox="0 0 1280 720"
          style={{
            position: 'absolute',
            width: '1280px',
            height: '720px',
            top: 0,
            left: 0,
            zIndex: 10,
            pointerEvents: 'none',
          }}
        >
          {leaf_nodes.map((_, idx) => {
            const angle = getLeafAngle(idx, N_leaves);
            const x1 = centerX;
            const y1 = centerY;
            const x2 = centerX + radius * Math.cos(angle);
            const y2 = centerY + radius * Math.sin(angle);

            // Animate lines drawing outwards
            const currentX = interpolate(lineProgress, [0, 1], [x1, x2]);
            const currentY = interpolate(lineProgress, [0, 1], [y1, y2]);

            return (
              <line
                key={`line-${idx}`}
                x1={x1}
                y1={y1}
                x2={currentX}
                y2={currentY}
                stroke={activeTheme.accentColor}
                strokeWidth="3.5"
                strokeLinecap="round"
                opacity={lineProgress > 0 ? 0.75 : 0}
                style={{
                  filter: `drop-shadow(0 0 4px ${activeTheme.accentColor})`,
                }}
              />
            );
          })}
        </svg>

        {/* Central Node */}
        <div
          style={{
            position: 'absolute',
            width: '150px',
            height: '150px',
            left: `${centerX}px`,
            top: `${centerY}px`,
            background: activeTheme.accentColor,
            color: '#090d16',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            textAlign: 'center',
            fontWeight: 800,
            fontSize: '20px',
            boxShadow: `0 12px 36px rgba(${activeTheme.accentColorRgb}, 0.5)`,
            transform: `translate(-50%, -50%) scale(${centerScale}) translateY(${floatY}px)`,
            zIndex: 20,
            padding: '18px',
            boxSizing: 'border-box',
            lineHeight: '1.2',
          }}
        >
          {central_node}
        </div>

        {/* Leaf Nodes */}
        {leaf_nodes.map((node, idx) => {
          const angle = getLeafAngle(idx, N_leaves);
          const x = centerX + radius * Math.cos(angle);
          const y = centerY + radius * Math.sin(angle);

          const scale = leafSprings[idx];
          const opacity = leafSprings[idx];

          return (
            <div
              key={`leaf-${idx}`}
              style={{
                position: 'absolute',
                left: `${x}px`,
                top: `${y}px`,
                padding: '14px 22px',
                background: 'rgba(15, 23, 42, 0.9)',
                border: `2px solid ${activeTheme.accentColor}`,
                borderRadius: '16px',
                color: '#ffffff',
                fontWeight: 700,
                fontSize: '16px',
                textAlign: 'center',
                boxShadow: '0 12px 28px rgba(0,0,0,0.3)',
                transform: `translate(-50%, -50%) scale(${scale})`,
                opacity,
                zIndex: 30,
                whiteSpace: 'nowrap',
                transition: 'border 0.3s ease',
              }}
            >
              {node}
            </div>
          );
        })}
      </div>
    </div>
  );
};
