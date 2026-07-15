import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, spring } from 'remotion';
import { getTheme } from '../themeHelper';
import katex from 'katex';
import 'katex/dist/katex.min.css';

const renderLaTeX = (latexStr: string) => {
  try {
    const clean = latexStr.trim().replace(/^\$+|\$+$/g, '');
    return katex.renderToString(clean, { throwOnError: false });
  } catch (e) {
    return latexStr;
  }
};

interface CoordPoint {
  x: number; // grid coordinate, e.g. -5 to 5
  y: number; // grid coordinate, e.g. -5 to 5
  label?: string;
}

interface ConnectionLine {
  from_idx: number;
  to_idx: number;
  label?: string;
}

interface CartesianGridProps {
  title: string;
  points: CoordPoint[];
  lines: ConnectionLine[];
  equation_label?: string;
  theme: string;
  svg_elements?: any[];
}

// Translate grid coordinates to SVG 1000x1000 pixel coordinates
const translateCoordToPixel = (val: number, isX: boolean): number => {
  // If the coordinate is already in pixel space (e.g. > 15 or < -15), leave it as is
  if (val > 15 || val < -15) {
    return val;
  }
  // Center (0,0) is 500. Grid span [-6, 6] (width 12) maps to [100, 900] (width 800)
  // pixel = 500 + grid_coord * (800 / 12) = 500 + grid_coord * 66.666
  // Y-axis is inverted in SVG
  return isX ? 500 + val * 66.666 : 500 - val * 66.666;
};

// Translate SVG path string from grid space to pixel space if needed
const translatePathToPixel = (d: string): string => {
  const numberRegex = /-?\d+(\.\d+)?/g;
  const matches = d.match(numberRegex);
  if (!matches) return d;

  const numbers = matches.map(Number);
  const isGridSpace = numbers.every(n => n >= -15 && n <= 15);
  if (!isGridSpace) {
    return d; // Already in pixel space
  }

  let isX = true;
  return d.replace(/-?\d+(\.\d+)?/g, (match) => {
    const val = Number(match);
    const translated = translateCoordToPixel(val, isX);
    isX = !isX;
    return translated.toFixed(1);
  });
};

export const CartesianGrid: React.FC<CartesianGridProps> = ({
  title,
  points = [],
  lines = [],
  equation_label,
  theme,
  svg_elements = [],
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const activeTheme = getTheme(theme);

  // Layout limits
  const minVal = -6;
  const maxVal = 6;
  const size = 320; // Canvas SVG width/height

  // Conversions from grid coordinate to SVG px
  const toSvgX = (gridX: number) => {
    return ((gridX - minVal) / (maxVal - minVal)) * size;
  };
  const toSvgY = (gridY: number) => {
    return size - ((gridY - minVal) / (maxVal - minVal)) * size;
  };

  // Entrance animations
  const gridOpacity = interpolate(frame, [0, 15], [0, 0.25], { extrapolateRight: 'clamp' });
  const axisProgress = spring({
    frame: frame - 10,
    fps,
    config: { stiffness: 90, damping: 15 }
  });

  // Points pop animation
  const pointsSprings = points.map((_, idx) => {
    return spring({
      frame: frame - (20 + idx * 8),
      fps,
      config: { stiffness: 130, damping: 12 }
    });
  });

  // Lines & Curve drawing progress
  const linesProgress = spring({
    frame: frame - 25,
    fps,
    config: { stiffness: 70, damping: 14 }
  });

  // Title fade-in
  const titleOpacity = interpolate(frame, [0, 15], [0, 1], { extrapolateRight: 'clamp' });
  const titleTranslateY = interpolate(frame, [0, 15], [-15, 0], { extrapolateRight: 'clamp' });

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        padding: '50px 60px',
        boxSizing: 'border-box',
        fontFamily: activeTheme.fontFamily,
        color: activeTheme.textColor,
        alignItems: 'center',
        justifyContent: 'space-between',
      }}
    >
      {/* Left side: Equations & Coordinates definitions */}
      <div
        style={{
          width: '42%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
        }}
      >
        <h2
          style={{
            fontSize: '34px',
            fontWeight: 800,
            margin: '0 0 20px 0',
            opacity: titleOpacity,
            transform: `translateY(${titleTranslateY}px)`,
            color: activeTheme.accentColor,
          }}
        >
          {title}
        </h2>

        {/* Equation Display Box */}
        {equation_label && (
          <div
            style={{
              padding: '16px 20px',
              background: 'rgba(15, 23, 42, 0.65)',
              border: `2px solid ${activeTheme.accentColor}`,
              borderRadius: '16px',
              fontSize: '22px',
              fontWeight: 700,
              marginBottom: '24px',
              boxShadow: `0 8px 24px rgba(${activeTheme.accentColorRgb}, 0.2)`,
              alignSelf: 'flex-start',
              opacity: titleOpacity,
              color: '#ffffff',
            }}
            dangerouslySetInnerHTML={{ __html: renderLaTeX(equation_label) }}
          />
        )}

        {/* Coordinate Points details */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {points.map((p, idx) => {
            const scale = pointsSprings[idx];
            return (
              <div
                key={`point-info-${idx}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  fontSize: '15px',
                  fontWeight: 600,
                  opacity: scale,
                  transform: `translateX(${interpolate(scale, [0, 1], [-15, 0])}px)`,
                }}
              >
                <div
                  style={{
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    background: activeTheme.accentColor,
                    marginRight: '12px',
                    boxShadow: `0 0 8px ${activeTheme.accentColor}`,
                  }}
                />
                <span style={{ color: activeTheme.accentColor, marginRight: '8px' }}>
                  {p.label || `P${idx + 1}`} :
                </span>
                <span>({p.x}, {p.y})</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Right side: Interactive Cartesian Grid Canvas */}
      <div
        style={{
          width: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <div
          style={{
            position: 'relative',
            width: `${size}px`,
            height: `${size}px`,
            background: 'rgba(15, 23, 42, 0.4)',
            border: '2px solid rgba(255,255,255,0.06)',
            borderRadius: '24px',
            boxShadow: '0 16px 40px rgba(0,0,0,0.3)',
            boxSizing: 'border-box',
            overflow: 'hidden',
          }}
        >
          {/* Base Grid Layer (320x320) */}
          <svg
            viewBox={`0 0 ${size} ${size}`}
            style={{
              position: 'absolute',
              width: '100%',
              height: '100%',
              top: 0,
              left: 0,
            }}
          >
            {/* Grid Lines */}
            <g opacity={gridOpacity}>
              {Array.from({ length: maxVal - minVal + 1 }).map((_, idx) => {
                const val = minVal + idx;
                const svgPos = toSvgX(val);
                return (
                  <React.Fragment key={`grid-${val}`}>
                    <line x1={svgPos} y1="0" x2={svgPos} y2={size} stroke="#ffffff" strokeWidth="0.5" />
                    <line x1="0" y1={svgPos} x2={size} y2={svgPos} stroke="#ffffff" strokeWidth="0.5" />
                  </React.Fragment>
                );
              })}
            </g>

            {/* X Axis and Y Axis */}
            {axisProgress > 0 && (
              <g opacity={0.6}>
                <line
                  x1={toSvgX(minVal)}
                  y1={toSvgY(0)}
                  x2={interpolate(axisProgress, [0, 1], [toSvgX(minVal), toSvgX(maxVal)])}
                  y2={toSvgY(0)}
                  stroke="#ffffff"
                  strokeWidth="2.5"
                />
                <line
                  x1={toSvgX(0)}
                  y1={toSvgY(minVal)}
                  x2={toSvgX(0)}
                  y2={interpolate(axisProgress, [0, 1], [toSvgY(minVal), toSvgY(maxVal)])}
                  stroke="#ffffff"
                  strokeWidth="2.5"
                />
              </g>
            )}

            {/* Connecting lines between coordinate points */}
            {linesProgress > 0 &&
              lines.map((line, idx) => {
                const p1 = points[line.from_idx];
                const p2 = points[line.to_idx];
                if (!p1 || !p2) return null;

                const x1 = toSvgX(p1.x);
                const y1 = toSvgY(p1.y);
                const x2 = toSvgX(p2.x);
                const y2 = toSvgY(p2.y);

                const currentX = interpolate(linesProgress, [0, 1], [x1, x2]) as unknown as number;
                const currentY = interpolate(linesProgress, [0, 1], [y1, y2]) as unknown as number;

                return (
                  <line
                    key={`line-connect-${idx}`}
                    x1={x1}
                    y1={y1}
                    x2={currentX}
                    y2={currentY}
                    stroke={activeTheme.accentColor}
                    strokeWidth="3"
                    style={{
                      filter: `drop-shadow(0 0 4px ${activeTheme.accentColor})`,
                    }}
                  />
                );
              })}

            {/* Plotted coordinate points */}
            {points.map((p, idx) => {
              const scale = pointsSprings[idx];
              if (scale <= 0) return null;

              const svgX = toSvgX(p.x);
              const svgY = toSvgY(p.y);

              return (
                <g key={`point-dot-${idx}`} transform={`translate(${svgX}, ${svgY}) scale(${scale})`}>
                  <circle cx="0" cy="0" r="10" fill={activeTheme.accentColor} opacity="0.3" />
                  <circle cx="0" cy="0" r="5" fill={activeTheme.accentColor} />
                  <text
                    x="8"
                    y="-8"
                    fill="#ffffff"
                    fontSize="11"
                    fontWeight="700"
                    style={{
                      textShadow: '1px 1px 4px #000',
                    }}
                  >
                    {p.label || `(${p.x},${p.y})`}
                  </text>
                </g>
              );
            })}
          </svg>

          {/* Upgraded Layer: Custom SVG Elements Overlay (1000x1000 viewBox matching LLM coordinate output) */}
          {svg_elements && svg_elements.length > 0 && (
            <svg
              viewBox="0 0 1000 1000"
              style={{
                position: 'absolute',
                width: '100%',
                height: '100%',
                top: 0,
                left: 0,
                zIndex: 15,
                pointerEvents: 'none',
              }}
            >
              {svg_elements.map((el, elIdx) => {
                const strokeColor = el.stroke || activeTheme.accentColor;
                const fillColor = el.fill && el.fill !== 'none' ? el.fill : 'none';
                const strokeWidth = el.stroke_width || 3;

                // Animate path drawing (like drawing a parabola curve) using linesProgress
                if (el.type === 'path') {
                  const translatedPath = translatePathToPixel(el.d);
                  const pathLength = 1500; // safe estimation for curve path perimeter
                  const strokeDashoffset = pathLength * (1 - linesProgress);

                  return (
                    <path
                      key={`custom-path-${elIdx}`}
                      d={translatedPath}
                      fill={fillColor}
                      stroke={strokeColor}
                      strokeWidth={strokeWidth}
                      strokeLinecap="round"
                      strokeDasharray={pathLength}
                      strokeDashoffset={strokeDashoffset}
                      style={{
                        filter: `drop-shadow(0 0 6px ${strokeColor})`,
                        transition: 'stroke 0.4s ease',
                      }}
                    />
                  );
                }

                // Animate custom straight lines drawing
                if (el.type === 'line') {
                  const x1 = translateCoordToPixel(el.x1 || 0, true);
                  const y1 = translateCoordToPixel(el.y1 || 0, false);
                  const x2 = translateCoordToPixel(el.x2 || 0, true);
                  const y2 = translateCoordToPixel(el.y2 || 0, false);

                  const currentX = interpolate(linesProgress, [0, 1], [x1, x2]) as unknown as number;
                  const currentY = interpolate(linesProgress, [0, 1], [y1, y2]) as unknown as number;

                  return (
                    <line
                      key={`custom-line-${elIdx}`}
                      x1={x1}
                      y1={y1}
                      x2={currentX}
                      y2={currentY}
                      stroke={strokeColor}
                      strokeWidth={strokeWidth}
                    />
                  );
                }

                // Animate custom circles popping in (like coordinates roots)
                if (el.type === 'circle') {
                  const scale = spring({
                    frame: frame - (30 + elIdx * 10),
                    fps,
                    config: { stiffness: 120, damping: 14 }
                  });

                  const cx = translateCoordToPixel(el.cx || 0, true);
                  const cy = translateCoordToPixel(el.cy || 0, false);

                  return (
                    <g
                      key={`custom-circle-${elIdx}`}
                      transform={`translate(${cx}, ${cy}) scale(${scale})`}
                    >
                      <circle cx="0" cy="0" r={el.r || 12} fill={strokeColor} opacity="0.3" />
                      <circle cx="0" cy="0" r={(el.r || 12) * 0.6} fill={strokeColor} />
                      {el.label && (
                        <text
                          x="15"
                          y="5"
                          fill="#ffffff"
                          fontSize="30"
                          fontWeight="700"
                          style={{
                            textShadow: '2px 2px 6px #000',
                          }}
                        >
                          {el.label}
                        </text>
                      )}
                    </g>
                  );
                }

                return null;
              })}
            </svg>
          )}
        </div>
      </div>
    </div>
  );
};
