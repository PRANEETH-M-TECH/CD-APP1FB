import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, spring } from 'remotion';
import { getTheme } from '../themeHelper';

interface GeoMarkerPoint {
  x: number; // 0 to 100 percentage position
  y: number; // 0 to 100 percentage position
  label: string;
}

interface GeoMarkerProps {
  title: string;
  map_name?: string;
  markers: GeoMarkerPoint[];
  theme: string;
}

export const GeoMarker: React.FC<GeoMarkerProps> = ({
  title,
  map_name = 'india',
  markers = [],
  theme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const activeTheme = getTheme(theme);

  // Title fade in
  const titleOpacity = interpolate(frame, [0, 15], [0, 1], { extrapolateRight: 'clamp' });
  const titleTranslateY = interpolate(frame, [0, 15], [-15, 0], { extrapolateRight: 'clamp' });

  // Map fade in & scale
  const mapSpring = spring({
    frame,
    fps,
    config: { stiffness: 90, damping: 15 }
  });
  const mapScale = interpolate(mapSpring, [0, 1], [0.85, 1.0]);
  const mapOpacity = interpolate(frame, [5, 20], [0, 0.95], { extrapolateRight: 'clamp' });

  // Markers staggered entry springs
  const markerSprings = markers.map((_, idx) => {
    return spring({
      frame: frame - (20 + idx * 10),
      fps,
      config: { stiffness: 120, damping: 14 }
    });
  });

  // Pulse sonar ring loop
  const pulseScale = interpolate((frame % 30), [0, 30], [1, 2.5]);
  const pulseOpacity = interpolate((frame % 30), [0, 30], [0.8, 0]);

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
      {/* Left Column: Title & Marker Description Cards */}
      <div
        style={{
          width: '40%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
        }}
      >
        <h2
          style={{
            fontSize: '34px',
            fontWeight: 800,
            margin: '0 0 24px 0',
            opacity: titleOpacity,
            transform: `translateY(${titleTranslateY}px)`,
            color: activeTheme.accentColor,
          }}
        >
          {title}
        </h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {markers.map((marker, idx) => {
            const scale = markerSprings[idx];
            return (
              <div
                key={`marker-info-${idx}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  background: activeTheme.cardBackground,
                  border: activeTheme.cardBorder,
                  borderRadius: '12px',
                  padding: '12px 16px',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                  transform: `scale(${scale})`,
                  opacity: scale,
                }}
              >
                <div
                  style={{
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    background: activeTheme.accentColor,
                    marginRight: '14px',
                    boxShadow: `0 0 8px ${activeTheme.accentColor}`,
                  }}
                />
                <span style={{ fontWeight: 700, fontSize: '16px' }}>{marker.label}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Right Column: Map outline canvas */}
      <div
        style={{
          width: '54%',
          height: '100%',
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {/* Map visual shell */}
        <div
          style={{
            width: '340px',
            height: '380px',
            position: 'relative',
            background: 'rgba(15, 23, 42, 0.45)',
            border: '2px solid rgba(255,255,255,0.06)',
            borderRadius: '24px',
            boxShadow: '0 16px 40px rgba(0,0,0,0.25)',
            overflow: 'hidden',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transform: `scale(${mapScale})`,
            opacity: mapOpacity,
          }}
        >
          {/* Simple Vector India Map SVG Outline */}
          <svg
            viewBox="0 0 200 220"
            style={{
              width: '85%',
              height: '85%',
              stroke: activeTheme.accentColor,
              strokeWidth: '1.5',
              fill: 'rgba(255, 255, 255, 0.02)',
              opacity: 0.75,
            }}
          >
            {/* Outline of India placeholder path */}
            <path d="M100 20 L110 35 L120 30 L115 45 L130 50 L125 65 L145 70 L140 85 L130 90 L135 110 L150 115 L140 135 L125 130 L120 145 L130 165 L115 170 L105 195 L100 210 L95 195 L85 170 L70 165 L80 145 L75 130 L60 135 L50 115 L65 110 L70 90 L60 85 L55 70 L75 65 L70 50 L85 45 L80 30 L90 35 Z" />
          </svg>

          {/* Coordinate Map Markers */}
          {markers.map((marker, idx) => {
            const scale = markerSprings[idx];
            if (scale <= 0) return null;

            // Map coordinates relative to 340x380 container size
            const posX = (marker.x / 100) * 340;
            const posY = (marker.y / 100) * 380;

            return (
              <div
                key={`map-pin-${idx}`}
                style={{
                  position: 'absolute',
                  left: `${posX}px`,
                  top: `${posY}px`,
                  transform: `translate(-50%, -50%) scale(${scale})`,
                  zIndex: 20,
                }}
              >
                {/* Sonar Ripple Rings */}
                <div
                  style={{
                    position: 'absolute',
                    width: '32px',
                    height: '32px',
                    left: '-11px',
                    top: '-11px',
                    borderRadius: '50%',
                    border: `2px solid ${activeTheme.accentColor}`,
                    transform: `scale(${pulseScale})`,
                    opacity: pulseOpacity,
                  }}
                />

                {/* Central pin dot */}
                <div
                  style={{
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    background: '#ffffff',
                    border: `3.5px solid ${activeTheme.accentColor}`,
                    boxShadow: `0 0 10px ${activeTheme.accentColor}`,
                  }}
                />

                {/* Micro Label */}
                <div
                  style={{
                    position: 'absolute',
                    left: '14px',
                    top: '-6px',
                    background: 'rgba(15, 23, 42, 0.9)',
                    border: '1px solid rgba(255,255,255,0.15)',
                    borderRadius: '6px',
                    padding: '3px 8px',
                    fontSize: '11px',
                    fontWeight: 700,
                    whiteSpace: 'nowrap',
                    boxShadow: '0 2px 6px rgba(0,0,0,0.3)',
                  }}
                >
                  {marker.label}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
