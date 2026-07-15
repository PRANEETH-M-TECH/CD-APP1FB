import React, { useMemo } from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, Easing, Img, staticFile } from 'remotion';
import { getPointAtLength, getLength } from '@remotion/paths';
import { ZoomTarget, ImageAnnotation, MotionPath, SpotlightRegion } from '../types';

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

interface ImageSceneProps {
  title: string;
  teacher_script: string;
  image_url: string;
  zoom_targets?: ZoomTarget[];
  annotations?: ImageAnnotation[];
  motion_path?: MotionPath;
  spotlight?: SpotlightRegion;
  animation_style: 'zoom_and_annotate' | 'spotlight_reveal' | 'motion_path' | 'progressive_reveal' | 'simple_zoom';
  theme: 'indigo' | 'gold' | 'emerald' | 'rose';
}

function interpolateKeyframes(
  targets: ZoomTarget[],
  currentPercent: number,
  durationInFrames: number
) {
  if (!targets || targets.length === 0) {
    // Default gentle zoom
    const scale = interpolate(currentPercent, [0, 100], [1, 1.12], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
      easing: Easing.bezier(0.25, 0.1, 0.25, 1.0),
    });
    return { scale, x: 50, y: 50 };
  }

  // Sort keyframes by time percentage
  const sorted = [...targets].sort((a, b) => a.at_percent - b.at_percent);

  // Pad start if needed
  if (sorted[0].at_percent > 0) {
    sorted.unshift({ ...sorted[0], at_percent: 0, scale: 1.0, x: 50, y: 50 });
  }
  // Pad end if needed
  if (sorted[sorted.length - 1].at_percent < 100) {
    sorted.push({ ...sorted[sorted.length - 1], at_percent: 100 });
  }

  // If before first target
  if (currentPercent <= sorted[0].at_percent) {
    return { scale: sorted[0].scale, x: sorted[0].x, y: sorted[0].y };
  }
  // If after last target
  if (currentPercent >= sorted[sorted.length - 1].at_percent) {
    const last = sorted[sorted.length - 1];
    return { scale: last.scale, x: last.x, y: last.y };
  }

  // Find current active segment
  let idx = 0;
  for (let i = 0; i < sorted.length - 1; i++) {
    if (currentPercent >= sorted[i].at_percent && currentPercent <= sorted[i + 1].at_percent) {
      idx = i;
      break;
    }
  }

  const kf1 = sorted[idx];
  const kf2 = sorted[idx + 1];

  const segmentProgress = (currentPercent - kf1.at_percent) / (kf2.at_percent - kf1.at_percent);
  // Apply a smooth easing curve for panning/zooming transitions
  const easedProgress = Easing.bezier(0.25, 0.1, 0.25, 1.0)(segmentProgress);

  const scale = interpolate(easedProgress, [0, 1], [kf1.scale, kf2.scale]);
  const x = interpolate(easedProgress, [0, 1], [kf1.x, kf2.x]);
  const y = interpolate(easedProgress, [0, 1], [kf1.y, kf2.y]);

  return { scale, x, y };
}

export const ImageScene: React.FC<ImageSceneProps> = ({
  title,
  teacher_script,
  image_url,
  zoom_targets = [],
  annotations = [],
  motion_path,
  spotlight,
  animation_style,
  theme,
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const currentPercent = (frame / durationInFrames) * 100;

  const accentColor = THEME_ACCENTS[theme] || THEME_ACCENTS.indigo;
  const accentRgb = THEME_ACCENT_RGBS[theme] || THEME_ACCENT_RGBS.indigo;

  // Scene transition: 15 frame fade-in at start, 15 frame fade-out at end
  const sceneOpacity = interpolate(
    frame,
    [0, 15, durationInFrames - 15, durationInFrames],
    [0, 1, 1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  // Compute Zoom / Pan values
  const { scale, x, y } = useMemo(() => {
    return interpolateKeyframes(zoom_targets, currentPercent, durationInFrames);
  }, [zoom_targets, currentPercent, durationInFrames]);

  // Calculate translation values to center (x, y) coordinates with screen boundary clamping
  const maxTx = 50 * (scale - 1);
  const minTx = -maxTx;
  const tx = scale > 1 ? Math.max(minTx, Math.min(maxTx, -(x - 50) * scale)) : 0;

  const maxTy = 50 * (scale - 1);
  const minTy = -maxTy;
  const ty = scale > 1 ? Math.max(minTy, Math.min(maxTy, -(y - 50) * scale)) : 0;

  // Spotlight progress
  const spotlightOpacity = useMemo(() => {
    if (!spotlight) return 0;
    const spotlightProgress = (currentPercent - spotlight.at_percent) / 10;
    return interpolate(spotlightProgress, [0, 1], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
  }, [spotlight, currentPercent]);

  // Motion path dot position
  const dotPos = useMemo(() => {
    if (!motion_path || !motion_path.path_data) return null;
    try {
      const length = getLength(motion_path.path_data);
      if (length === 0) return null;

      const startP = motion_path.start_percent ?? 10;
      const durationP = motion_path.duration_percent ?? 70;
      const progress = (currentPercent - startP) / durationP;
      const clampedProgress = Math.max(0, Math.min(1, progress));

      // Interpolate along the path
      if (clampedProgress > 0) {
        return getPointAtLength(motion_path.path_data, clampedProgress * length);
      }
    } catch (e) {
      console.warn('Failed to calculate motion path coordinates:', e);
    }
    return null;
  }, [motion_path, currentPercent]);

  // Log active state for preview and telemetry
  React.useEffect(() => {
    if (frame === 0) {
      console.log(`[ImageScene] Mounted scene: "${title}"`);
      console.log(` - Image URL: ${image_url}`);
      console.log(` - Animation Style: ${animation_style}`);
      console.log(` - Zoom Targets: ${zoom_targets.length}`);
      console.log(` - Annotations: ${annotations.length}`);
      if (motion_path) console.log(' - Motion Path Configured');
      if (spotlight) console.log(` - Spotlight at percent ${spotlight.at_percent}%`);
    }
  }, [frame, title, image_url, animation_style, zoom_targets, annotations, motion_path, spotlight]);

  // Fallback view if no image is provided
  if (!image_url) {
    return (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'linear-gradient(135deg, #111827 0%, #1f2937 100%)',
          color: '#ffffff',
          opacity: sceneOpacity,
          padding: '40px',
          boxSizing: 'border-box',
        }}
      >
        <div
          style={{
            background: 'rgba(255, 255, 255, 0.05)',
            border: '1px dashed rgba(255, 255, 255, 0.15)',
            borderRadius: '16px',
            padding: '40px',
            textAlign: 'center',
            maxWidth: '600px',
          }}
        >
          <span style={{ fontSize: '48px', display: 'block', marginBottom: '16px' }}>📷</span>
          <h2 style={{ fontSize: '24px', fontWeight: 600, margin: '0 0 12px 0' }}>{title}</h2>
          <p style={{ fontSize: '16px', color: 'rgba(255,255,255,0.6)', margin: 0, lineHeight: 1.5 }}>
            No image asset supplied. Video will present subtitle narration.
          </p>
        </div>
        {/* Teacher Narration Subtitles */}
        {teacher_script && (
          <div
            style={{
              position: 'absolute',
              bottom: 45,
              left: '8%',
              right: '8%',
              textAlign: 'center',
              fontSize: '22px',
              fontWeight: 700,
              lineHeight: '1.4',
              color: '#ffffff',
              zIndex: 90,
              textShadow: '0 2px 4px rgba(0,0,0,0.9), 0 0 10px rgba(0,0,0,0.9), 1px 1px 0px #000, -1px -1px 0px #000, 1px -1px 0px #000, -1px 1px 0px #000',
              fontFamily: 'Inter, system-ui, sans-serif',
            }}
          >
            {teacher_script}
          </div>
        )}
      </div>
    );
  }

  const imageSrc = image_url.startsWith('http') || image_url.startsWith('data:') ? image_url : staticFile(image_url);

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        overflow: 'hidden',
        opacity: sceneOpacity,
        background: '#020617', // Black border background
      }}
    >
      {/* 1. Zoomed / Panned Image Layer */}
      <div
        style={{
          width: '100%',
          height: '100%',
          position: 'absolute',
          transform: `scale(${scale}) translate(${tx}%, ${ty}%)`,
          transformOrigin: '50% 50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Img
          src={imageSrc}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            filter: spotlight ? `brightness(${1 - 0.7 * spotlightOpacity})` : undefined,
          }}
        />

        {/* 2. Spotlight Mask Overlay */}
        {spotlight && (
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              clipPath: `circle(${spotlight.radius}px at ${spotlight.x}% ${spotlight.y}%)`,
              opacity: spotlightOpacity,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              pointerEvents: 'none',
            }}
          >
            <Img
              src={imageSrc}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'contain',
              }}
            />
          </div>
        )}

        {/* 3. SVG Annotations (Arrows and Circles) */}
        {annotations.length > 0 && (
          <svg
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              pointerEvents: 'none',
              zIndex: 10,
            }}
          >
            <defs>
              {annotations.map((ann, idx) => {
                if (ann.type !== 'arrow') return null;
                return (
                  <marker
                    key={`arrowhead-${idx}`}
                    id={`arrowhead-${idx}`}
                    markerWidth="10"
                    markerHeight="7"
                    refX="6"
                    refY="3.5"
                    orient="auto"
                  >
                    <polygon points="0 0, 10 3.5, 0 7" fill={ann.color || accentColor} />
                  </marker>
                );
              })}
            </defs>

            {annotations.map((ann, idx) => {
              if (currentPercent < ann.at_percent) return null;

              // Animate annotation in over a short interval (e.g. 6% of timeline)
              const annProgress = Math.max(0, Math.min(1, (currentPercent - ann.at_percent) / 6));
              const easedProgress = Easing.bezier(0.16, 1, 0.3, 1)(annProgress);

              if (ann.type === 'circle') {
                return (
                  <g key={`ann-circle-${idx}`}>
                    {/* Pulsing Outer Ring */}
                    <circle
                      cx={`${ann.x}%`}
                      cy={`${ann.y}%`}
                      r={easedProgress * 25}
                      fill="none"
                      stroke={ann.color || accentColor}
                      strokeWidth={2}
                      opacity={1 - easedProgress}
                    />
                    {/* Inner Core */}
                    <circle
                      cx={`${ann.x}%`}
                      cy={`${ann.y}%`}
                      r={6 * Math.min(1, easedProgress * 1.5)}
                      fill={ann.color || accentColor}
                      opacity={Math.min(1, easedProgress * 1.5)}
                    />
                  </g>
                );
              }

              if (ann.type === 'arrow' && ann.target_x !== undefined && ann.target_y !== undefined) {
                const curX = ann.x + (ann.target_x - ann.x) * easedProgress;
                const curY = ann.y + (ann.target_y - ann.y) * easedProgress;
                return (
                  <line
                    key={`ann-arrow-${idx}`}
                    x1={`${ann.x}%`}
                    y1={`${ann.y}%`}
                    x2={`${curX}%`}
                    y2={`${curY}%`}
                    stroke={ann.color || accentColor}
                    strokeWidth={3}
                    markerEnd={`url(#arrowhead-${idx})`}
                    opacity={easedProgress}
                  />
                );
              }

              return null;
            })}
          </svg>
        )}

        {/* 4. Text Labels Layer (position overlay) */}
        {annotations.map((ann, idx) => {
          if (ann.type !== 'label' || currentPercent < ann.at_percent) return null;
          const annProgress = Math.max(0, Math.min(1, (currentPercent - ann.at_percent) / 6));
          const scaleIn = interpolate(annProgress, [0, 1], [0.85, 1]);
          const opacityIn = interpolate(annProgress, [0, 1], [0.1, 1]);

          return (
            <div
              key={`ann-label-${idx}`}
              style={{
                position: 'absolute',
                left: `${ann.x}%`,
                top: `${ann.y}%`,
                transform: `translate(-50%, -100%) scale(${scaleIn})`,
                opacity: opacityIn,
                backgroundColor: 'rgba(9, 13, 22, 0.88)',
                color: '#ffffff',
                padding: '6px 12px',
                borderRadius: '8px',
                fontSize: '14px',
                fontWeight: 600,
                border: `1.5px solid ${ann.color || accentColor}`,
                whiteSpace: 'nowrap',
                boxShadow: `0 8px 16px rgba(0, 0, 0, 0.5), 0 0 10px rgba(${accentRgb}, 0.25)`,
                zIndex: 20,
              }}
            >
              {ann.label}
            </div>
          );
        })}

        {/* 5. Motion Path SVG Guide & Dot Overlay */}
        {motion_path && (
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              pointerEvents: 'none',
              zIndex: 15,
            }}
          >
            <svg style={{ width: '100%', height: '100%' }}>
              <path
                d={motion_path.path_data}
                fill="none"
                stroke="rgba(255, 255, 255, 0.25)"
                strokeWidth={3}
                strokeDasharray="6,6"
              />
              {dotPos && (
                <circle
                  cx={dotPos.x}
                  cy={dotPos.y}
                  r={motion_path.dot_size ?? 8}
                  fill={motion_path.dot_color || '#ef4444'}
                  style={{
                    filter: `drop-shadow(0 0 8px ${motion_path.dot_color || '#ef4444'})`,
                  }}
                />
              )}
            </svg>
          </div>
        )}
      </div>

      {/* 6. Title floating card */}
      <div
        style={{
          position: 'absolute',
          top: 110,
          left: 40,
          background: 'rgba(10, 15, 30, 0.5)',
          backdropFilter: 'blur(8px)',
          padding: '10px 18px',
          borderRadius: '12px',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          zIndex: 80,
          pointerEvents: 'none',
        }}
      >
        <span
          style={{
            fontSize: '14px',
            textTransform: 'uppercase',
            letterSpacing: '1px',
            fontWeight: 700,
            color: accentColor,
          }}
        >
          {title}
        </span>
      </div>

      {/* Teacher Narration Subtitles */}
      {teacher_script && (
        <div
          style={{
            position: 'absolute',
            bottom: 45,
            left: '8%',
            right: '8%',
            textAlign: 'center',
            fontSize: '22px',
            fontWeight: 700,
            lineHeight: '1.4',
            color: '#ffffff',
            zIndex: 90,
            textShadow: '0 2px 4px rgba(0,0,0,0.9), 0 0 10px rgba(0,0,0,0.9), 1px 1px 0px #000, -1px -1px 0px #000, 1px -1px 0px #000, -1px 1px 0px #000',
            fontFamily: 'Inter, system-ui, sans-serif',
          }}
        >
          {teacher_script}
        </div>
      )}
    </div>
  );
};
