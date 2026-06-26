import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, Easing } from 'remotion';
import { SvgElement } from '../types';

interface DynamicIllustrationProps {
  title: string;
  svg_elements: SvgElement[];
  animation_action: 'rise' | 'fall' | 'spin' | 'scale_up' | 'slide_left' | 'slide_right' | 'none';
  theme: 'indigo' | 'gold' | 'emerald' | 'rose';
  canvas_color?: string;
}

const THEME_ACCENTS: Record<string, string> = {
  indigo: '#6366f1',
  gold: '#fbbf24',
  emerald: '#10b981',
  rose: '#f43f5e',
};

// Estimate the perimeter/stroke length for each element type
function estimateStrokeLength(el: SvgElement): number {
  switch (el.type) {
    case 'circle':
      return 2 * Math.PI * (el.r || 20);
    case 'ellipse':
      // Ramanujan's approximation for ellipse perimeter
      const a = el.rx || 20;
      const b = el.ry || 15;
      return Math.PI * (3 * (a + b) - Math.sqrt((3 * a + b) * (a + 3 * b)));
    case 'rect': {
      const w = el.width || 40;
      const h = el.height || 30;
      return 2 * (w + h);
    }
    case 'line': {
      const dx = (el.x2 || 0) - (el.x1 || 0);
      const dy = (el.y2 || 0) - (el.y1 || 0);
      return Math.sqrt(dx * dx + dy * dy);
    }
    case 'path':
      return 600; // safe estimate for most simple paths
    default:
      return 200;
  }
}

// Get the action animation transform for elements marked with animate=true
function getActionTransform(
  action: string,
  frame: number,
  actionStart: number,
  actionEnd: number,
): string {
  switch (action) {
    case 'rise': {
      const translateY = interpolate(frame, [actionStart, actionEnd], [0, -60], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
        easing: Easing.bezier(0.25, 0.1, 0.25, 1),
      });
      return `translateY(${translateY}px)`;
    }
    case 'fall': {
      const translateY = interpolate(frame, [actionStart, actionEnd], [0, 80], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
        easing: Easing.bezier(0.6, -0.28, 0.735, 0.045),
      });
      return `translateY(${translateY}px)`;
    }
    case 'spin': {
      const rotation = interpolate(frame, [actionStart, actionEnd], [0, 360], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      });
      return `rotate(${rotation}deg)`;
    }
    case 'scale_up': {
      const scale = interpolate(frame, [actionStart, actionEnd], [1, 1.4], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
        easing: Easing.bezier(0.16, 1, 0.3, 1),
      });
      return `scale(${scale})`;
    }
    case 'slide_left': {
      const translateX = interpolate(frame, [actionStart, actionEnd], [0, -100], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      });
      return `translateX(${translateX}px)`;
    }
    case 'slide_right': {
      const translateX = interpolate(frame, [actionStart, actionEnd], [0, 100], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      });
      return `translateX(${translateX}px)`;
    }
    default:
      return '';
  }
}

export const DynamicIllustration: React.FC<DynamicIllustrationProps> = ({
  title,
  svg_elements = [],
  animation_action = 'none',
  theme,
  canvas_color,
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const accentColor = THEME_ACCENTS[theme] || THEME_ACCENTS.indigo;

  // Phase timings (relative to scene duration)
  const drawProgress = interpolate(frame, [0, durationInFrames * 0.25], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const fillOpacity = interpolate(frame, [durationInFrames * 0.25, durationInFrames * 0.45], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const actionStart = durationInFrames * 0.50;
  const actionEnd = durationInFrames * 0.85;

  // Title fade in
  const titleOpacity = interpolate(frame, [0, 20], [0, 1], { extrapolateRight: 'clamp' });

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
          fontSize: '34px',
          fontWeight: 800,
          color: '#ffffff',
          textAlign: 'center',
          margin: '0 0 30px 0',
          textTransform: 'uppercase',
          letterSpacing: '1px',
          textShadow: '0 4px 8px rgba(0,0,0,0.5)',
          opacity: titleOpacity,
        }}
      >
        {title}
      </h2>

      {/* SVG Canvas */}
      <div
        style={{
          width: '600px',
          height: '380px',
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: '16px',
          background: canvas_color || 'rgba(255,255,255,0.03)',
          border: `1px solid ${accentColor}22`,
          overflow: 'hidden',
        }}
      >
        <svg
          viewBox="0 0 500 400"
          style={{ width: '100%', height: '100%' }}
          xmlns="http://www.w3.org/2000/svg"
        >
          {svg_elements.map((el, idx) => {
            const strokeLen = estimateStrokeLength(el);
            const dashOffset = strokeLen * (1 - drawProgress);
            const shouldAnimate = el.animate && animation_action !== 'none';
            const actionTransform = shouldAnimate
              ? getActionTransform(animation_action, frame, actionStart, actionEnd)
              : '';

            const commonStrokeProps = {
              stroke: el.stroke || accentColor,
              strokeWidth: el.stroke_width || 2,
              strokeDasharray: strokeLen,
              strokeDashoffset: dashOffset,
            };

            const commonFillProps = {
              fill: el.fill || 'none',
              fillOpacity: el.fill === 'none' ? 0 : fillOpacity,
            };

            const wrapperStyle: React.CSSProperties = shouldAnimate
              ? {
                  transformOrigin: 'center',
                  transformBox: 'fill-box' as any,
                  transform: actionTransform,
                }
              : {};

            // Render the appropriate SVG element
            let svgNode: React.ReactNode = null;

            switch (el.type) {
              case 'circle':
                svgNode = (
                  <circle
                    cx={el.cx || 0}
                    cy={el.cy || 0}
                    r={el.r || 20}
                    {...commonStrokeProps}
                    {...commonFillProps}
                    style={wrapperStyle}
                  />
                );
                break;

              case 'rect':
                svgNode = (
                  <rect
                    x={el.x || 0}
                    y={el.y || 0}
                    width={el.width || 40}
                    height={el.height || 30}
                    rx={el.rx || 0}
                    {...commonStrokeProps}
                    {...commonFillProps}
                    style={wrapperStyle}
                  />
                );
                break;

              case 'ellipse':
                svgNode = (
                  <ellipse
                    cx={el.cx || 0}
                    cy={el.cy || 0}
                    rx={el.rx || 30}
                    ry={el.ry || 20}
                    {...commonStrokeProps}
                    {...commonFillProps}
                    style={wrapperStyle}
                  />
                );
                break;

              case 'line':
                svgNode = (
                  <line
                    x1={el.x1 || 0}
                    y1={el.y1 || 0}
                    x2={el.x2 || 100}
                    y2={el.y2 || 100}
                    {...commonStrokeProps}
                    style={wrapperStyle}
                  />
                );
                break;

              case 'path':
                svgNode = (
                  <path
                    d={el.d || ''}
                    {...commonStrokeProps}
                    {...commonFillProps}
                    style={wrapperStyle}
                  />
                );
                break;
            }

            return (
              <g key={`svg-el-${idx}`}>
                {svgNode}
                {/* Optional text label */}
                {el.label && (
                  <text
                    x={el.cx || el.x || (el.x1 && el.x2 ? (el.x1 + el.x2) / 2 : 0)}
                    y={(el.cy || el.y || (el.y1 && el.y2 ? (el.y1 + el.y2) / 2 : 0)) + (el.r || el.height || 30) + 18}
                    textAnchor="middle"
                    fill="#e2e8f0"
                    fontSize="13"
                    fontWeight="600"
                    fontFamily="Inter, system-ui, sans-serif"
                    opacity={fillOpacity}
                  >
                    {el.label}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
};
