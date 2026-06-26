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

interface ConceptDiagramProps {
  left_title: string;
  left_bullets: string[];
  central_node: string;
  leaf_nodes: string[];
  theme: 'indigo' | 'gold' | 'emerald' | 'rose';
}

export const ConceptDiagram: React.FC<ConceptDiagramProps> = ({
  left_title,
  left_bullets = [],
  central_node,
  leaf_nodes = [],
  theme,
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  const accentColor = THEME_ACCENTS[theme] || THEME_ACCENTS.indigo;
  const accentRgb = THEME_ACCENT_RGBS[theme] || THEME_ACCENT_RGBS.indigo;

  // Total bullets and leaves counts
  const N_bullets = left_bullets.length;
  const N_leaves = leaf_nodes.length;

  // --- Animation Timing Configuration ---
  const leftColStart = 0;
  const leftColDuration = 15;

  // Title/bullets fade in
  const titleOpacity = interpolate(frame, [leftColStart, leftColStart + leftColDuration], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const bulletsOpacity = left_bullets.map((_, idx) => {
    // Stagger bullet point appearance
    const start = leftColStart + 10 + idx * 15;
    const end = start + 12;
    return interpolate(frame, [start, end], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
  });

  // Central Node appears (starts after left title shows up)
  const centerNodeStart = 15;
  const centerNodeEnd = 30;
  const centerScale = interpolate(frame, [centerNodeStart, centerNodeEnd], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  // SVG lines draw outward from center to outer leaf nodes
  const linesStart = 30;
  const linesEnd = 50;
  const lineProgress = interpolate(frame, [linesStart, linesEnd], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Outer Leaf Nodes fade-in and scale-up (triggered after lines reach them)
  const leafNodesOpacity = leaf_nodes.map((_, idx) => {
    // Slightly stagger the leaf nodes
    const start = linesEnd + idx * 6;
    const end = start + 12;
    return interpolate(frame, [start, end], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
  });

  // Float effect for the nodes to keep the diagram feeling alive
  const floatY = Math.sin((frame / 45) * Math.PI) * 4;

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        padding: '50px 60px',
        boxSizing: 'border-box',
        alignItems: 'center',
      }}
    >
      {/* Left Column: Text Summary */}
      <div
        style={{
          width: '40%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          paddingRight: '30px',
          boxSizing: 'border-box',
        }}
      >
        <h2
          style={{
            fontSize: '36px',
            fontWeight: 800,
            color: accentColor,
            margin: '0 0 20px 0',
            opacity: titleOpacity,
            textShadow: '0 2px 4px rgba(0,0,0,0.5)',
            textTransform: 'uppercase',
            letterSpacing: '-0.5px',
          }}
        >
          {left_title}
        </h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {left_bullets.map((bullet, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: '12px',
                opacity: bulletsOpacity[idx] ?? 0,
                transform: `translateX(${interpolate(frame, [10 + idx * 15, 10 + idx * 15 + 12], [-20, 0], { extrapolateRight: 'clamp' })}px)`,
              }}
            >
              <div
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  backgroundColor: accentColor,
                  marginTop: '10px',
                  boxShadow: `0 0 8px ${accentColor}`,
                  flexShrink: 0,
                }}
              />
              <p
                style={{
                  fontSize: '18px',
                  lineHeight: '1.5',
                  color: '#e2e8f0',
                  margin: 0,
                  fontWeight: 500,
                }}
              >
                {bullet}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Right Column: Node Diagram */}
      <div
        style={{
          width: '60%',
          height: '100%',
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxSizing: 'border-box',
        }}
      >
        {/* Draw Connection Lines in the background */}
        <svg
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            zIndex: 1,
          }}
        >
          {leaf_nodes.map((_, idx) => {
            const angle = (idx * 2 * Math.PI) / N_leaves - Math.PI / 2; // Start at top
            const radius = 170; // Node radius from center
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;

            // Animate lines drawing outwards from (50%, 50%)
            return (
              <line
                key={`line-${idx}`}
                x1="50%"
                y1="50%"
                x2={`calc(50% + ${x * lineProgress}px)`}
                y2={`calc(50% + ${y * (lineProgress + floatY * 0.002)}px)`}
                stroke={accentColor}
                strokeWidth="2.5"
                strokeDasharray="6,6"
                opacity={frame >= linesStart ? 0.6 : 0}
              />
            );
          })}
        </svg>

        {/* Central Node (Entity) */}
        <div
          style={{
            transform: `scale(${centerScale}) translateY(${floatY}px)`,
            background: `linear-gradient(135deg, #1e1b4b 0%, #0f0e26 100%)`,
            border: `3px solid ${accentColor}`,
            boxShadow: `0 8px 32px rgba(${accentRgb}, 0.25), 0 0 16px rgba(${accentRgb}, 0.1)`,
            borderRadius: '12px',
            padding: '22px 36px',
            color: '#ffffff',
            fontWeight: 800,
            fontSize: '24px',
            zIndex: 10,
            textAlign: 'center',
            minWidth: '150px',
            textTransform: 'uppercase',
            letterSpacing: '1px',
            textShadow: '0 2px 4px rgba(0,0,0,0.5)',
          }}
        >
          {central_node}
        </div>

        {/* Leaf Nodes (Attributes) */}
        {leaf_nodes.map((label, idx) => {
          const angle = (idx * 2 * Math.PI) / N_leaves - Math.PI / 2;
          const radius = 170;
          const x = Math.cos(angle) * radius;
          const y = Math.sin(angle) * radius;

          const scaleVal = leafNodesOpacity[idx] ?? 0;

          return (
            <div
              key={`leaf-${idx}`}
              style={{
                position: 'absolute',
                transform: `translate(calc(-50% + ${x}px), calc(-50% + ${y + floatY}px)) scale(${scaleVal})`,
                transformOrigin: 'center center',
                left: '50%',
                top: '50%',
                background: '#090d16',
                border: `2px solid rgba(255, 255, 255, 0.15)`,
                boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
                borderRadius: '24px',
                padding: '10px 20px',
                color: '#e2e8f0',
                fontSize: '14px',
                fontWeight: 700,
                zIndex: 12,
                whiteSpace: 'nowrap',
                letterSpacing: '0.5px',
                textTransform: 'uppercase',
                transition: 'border-color 0.2s',
              }}
            >
              {label}
            </div>
          );
        })}
      </div>
    </div>
  );
};
