import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, Easing } from 'remotion';
import { Asset } from './types';

// Standalone SVG path library for standard Lucide-style educational icons
const ICON_SVGS: Record<string, React.ReactNode> = {
  crown: <path d="M2 4l3 12h14l3-12-6 7-4-7-4 7-6-7zm3 16h14a1 1 0 0 1 1 1v1H4v-1a1 1 0 0 1 1-1z" />,
  shield: <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />,
  landmark: <path d="M3 22h18M6 18v-7m4 7v-7m4 7v-7m4 7v-7M2 11h20M12 2L2 7h20L12 2z" />,
  bank: <path d="M3 22h18M6 18v-7m4 7v-7m4 7v-7m4 7v-7M2 11h20M12 2L2 7h20L12 2z" />,
  users: (
    <>
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
      <path d="M16 3.13a4 4 0 0 1 0 7.75" />
    </>
  ),
  rupee: <path d="M6 3h12M6 8h12M6 3a6 6 0 0 1 6 6H6M6 9h12M14.5 14.5L6 21" />,
  clock: (
    <>
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </>
  ),
  bell: <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9M13.73 21a2 2 0 0 1-3.46 0" />,
  info: (
    <>
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </>
  ),
  book: (
    <>
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
    </>
  ),
  'book-open': (
    <>
      <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
      <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
    </>
  ),
  globe: (
    <>
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </>
  ),
  lock: (
    <>
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </>
  ),
  unlock: (
    <>
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
      <path d="M7 11V7a5 5 0 0 1 9.9-1" />
    </>
  ),
  landmark_building: <path d="M3 22h18M6 18v-7m4 7v-7m4 7v-7m4 7v-7M2 11h20M12 2L2 7h20L12 2z" />,
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </>
  ),
  arrow_right: <path d="M5 12h14M12 5l7 7-7 7" />,
  'arrow-right': <path d="M5 12h14M12 5l7 7-7 7" />,
  scroll: <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />,
  briefcase: (
    <>
      <rect x="2" y="7" width="20" height="14" rx="2" ry="2" />
      <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
    </>
  ),
  'map-pin': (
    <>
      <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
      <circle cx="12" cy="10" r="3" />
    </>
  ),
  handshake: (
    <>
      <path d="M11 18h2a2 2 0 0 0 2-2v-1a2 2 0 0 0-2-2h-2" />
      <path d="M18 10.5V7a2 2 0 0 0-2-2h-8a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h2" />
      <path d="M12 11h.01M16 11h.01" />
    </>
  ),
  'check-circle': (
    <>
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </>
  ),
  'x-circle': (
    <>
      <circle cx="12" cy="12" r="10" />
      <line x1="15" y1="9" x2="9" y2="15" />
      <line x1="9" y1="9" x2="15" y2="15" />
    </>
  ),
};

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

interface AssetViewProps {
  asset: Asset;
  theme: 'indigo' | 'gold' | 'emerald' | 'rose';
}

export const AssetView: React.FC<AssetViewProps> = ({ asset, theme }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const accentColor = THEME_ACCENTS[theme] || THEME_ACCENTS.indigo;
  const accentRgb = THEME_ACCENT_RGBS[theme] || THEME_ACCENT_RGBS.indigo;

  // 1. Calculate entrance animations
  let opacity = 1;
  let scale = 1;
  let translateX = 0;
  let rotate = 0;

  const hasFadeIn = asset.animations?.some((a) => a.type === 'fade_in');
  const hasScaleUp = asset.animations?.some((a) => a.type === 'scale_up');
  const hasSlideInLeft = asset.animations?.some((a) => a.type === 'slide_in_left');
  const hasSlideInRight = asset.animations?.some((a) => a.type === 'slide_in_right');

  if (hasFadeIn) opacity = 0;
  if (hasScaleUp) scale = 0;
  if (hasSlideInLeft) translateX = -100;
  if (hasSlideInRight) translateX = 100;

  asset.animations?.forEach((anim) => {
    const startFrame = anim.start_time * fps;
    const endFrame = startFrame + anim.duration * fps;

    if (anim.type === 'fade_in') {
      opacity = interpolate(frame, [startFrame, endFrame], [0, 1], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      });
    } else if (anim.type === 'fade_out') {
      opacity = interpolate(frame, [startFrame, endFrame], [1, 0], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      });
    } else if (anim.type === 'scale_up') {
      scale = interpolate(frame, [startFrame, endFrame], [0, 1], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
        easing: Easing.bezier(0.16, 1, 0.3, 1),
      });
    } else if (anim.type === 'scale_down') {
      scale = interpolate(frame, [startFrame, endFrame], [1, 0], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      });
    } else if (anim.type === 'slide_in_left') {
      translateX = interpolate(frame, [startFrame, endFrame], [-120, 0], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
        easing: Easing.bezier(0.16, 1, 0.3, 1),
      });
      opacity = interpolate(frame, [startFrame, startFrame + 8], [0, 1], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      });
    } else if (anim.type === 'slide_in_right') {
      translateX = interpolate(frame, [startFrame, endFrame], [120, 0], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
        easing: Easing.bezier(0.16, 1, 0.3, 1),
      });
      opacity = interpolate(frame, [startFrame, startFrame + 8], [0, 1], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      });
    } else if (anim.type === 'spin') {
      rotate = interpolate(frame, [startFrame, endFrame], [0, 360], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      });
    } else if (anim.type === 'appear') {
      opacity = frame >= startFrame ? 1 : 0;
    } else if (anim.type === 'disappear') {
      opacity = frame >= startFrame ? 0 : 1;
    }
  });

  // 2. Add subtle idle floating/bobbing for visual assets
  let bobbingY = 0;
  let floatingRotate = 0;
  if (asset.type === 'image' || asset.type === 'icon') {
    // Unique frequency/phase offset based on id hash to stagger different assets
    const phaseOffset = (asset.id.charCodeAt(0) || 0) * 0.1;
    bobbingY = Math.sin((frame / 30) * Math.PI * 0.8 + phaseOffset) * 4;
    floatingRotate = Math.sin((frame / 30) * Math.PI * 0.4 + phaseOffset) * 1;
  }

  // 3. Layout Styles
  const layoutStyle: React.CSSProperties = {
    position: 'absolute',
    top: `${asset.layout.top}%`,
    left: `${asset.layout.left}%`,
    width: `${asset.layout.width}%`,
    height: `${asset.layout.height}%`,
    opacity,
    transform: `translate(${translateX}px, ${bobbingY}px) scale(${scale}) rotate(${rotate + floatingRotate}deg)`,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    boxSizing: 'border-box',
  };

  // 4. Render by Type
  if (asset.type === 'text') {
    return (
      <div style={layoutStyle}>
        <div
          style={{
            color: '#f8fafc',
            fontSize: '24px',
            fontWeight: 800,
            textAlign: 'center',
            textShadow: '0 2px 10px rgba(0,0,0,0.5)',
            background: 'rgba(255, 255, 255, 0.05)',
            padding: '10px 20px',
            borderRadius: '12px',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            width: '100%',
            wordBreak: 'break-word',
          }}
        >
          {asset.text_content}
        </div>
      </div>
    );
  }

  if (asset.type === 'icon') {
    const iconName = (asset.search_query || 'info').toLowerCase();
    const svgContent = ICON_SVGS[iconName] || ICON_SVGS.info;

    return (
      <div style={layoutStyle}>
        <div
          style={{
            width: '100%',
            height: '100%',
            background: `radial-gradient(circle at 30% 30%, rgba(${accentRgb}, 0.2) 0%, rgba(${accentRgb}, 0.05) 100%)`,
            border: `2px solid ${accentColor}`,
            borderRadius: '50%',
            boxShadow: `0 8px 24px rgba(${accentRgb}, 0.2), inset 0 0 12px rgba(${accentRgb}, 0.1)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '15%',
            boxSizing: 'border-box',
          }}
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke={accentColor}
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{
              width: '100%',
              height: '100%',
              filter: `drop-shadow(0 0 4px ${accentColor})`,
            }}
          >
            {svgContent}
          </svg>
        </div>
        {asset.search_query && (
          <div
            style={{
              marginTop: '6px',
              fontSize: '11px',
              fontWeight: 600,
              color: 'rgba(255, 255, 255, 0.6)',
              textTransform: 'capitalize',
              letterSpacing: '0.5px',
              backgroundColor: 'rgba(0,0,0,0.3)',
              padding: '2px 8px',
              borderRadius: '6px',
            }}
          >
            {asset.search_query}
          </div>
        )}
      </div>
    );
  }

  if (asset.type === 'image') {
    const imageUrl = asset.asset_url;

    return (
      <div style={layoutStyle}>
        {imageUrl ? (
          <div
            style={{
              width: '100%',
              height: '100%',
              borderRadius: '16px',
              overflow: 'hidden',
              border: `2px solid rgba(${accentRgb}, 0.3)`,
              boxShadow: `0 12px 32px rgba(0, 0, 0, 0.4), 0 0 20px rgba(${accentRgb}, 0.1)`,
              position: 'relative',
              backgroundColor: '#0f172a',
            }}
          >
            <img
              src={imageUrl}
              alt={asset.search_query}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
              }}
              onError={(e) => {
                // If image fails to load, hide image and show query
                (e.target as HTMLElement).style.display = 'none';
              }}
            />
            {/* Soft overlay gradient */}
            <div
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: 'linear-gradient(to bottom, rgba(0,0,0,0) 60%, rgba(0,0,0,0.6) 100%)',
                pointerEvents: 'none',
              }}
            />
          </div>
        ) : (
          /* High-end decorative fallback card if Wikimedia query fails */
          <div
            style={{
              width: '100%',
              height: '100%',
              background: 'rgba(255, 255, 255, 0.03)',
              border: `2px dashed rgba(${accentRgb}, 0.4)`,
              borderRadius: '16px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '12px',
              boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
            }}
          >
            <div style={{ fontSize: '32px', marginBottom: '8px' }}>🖼️</div>
            <div
              style={{
                fontSize: '12px',
                fontWeight: 700,
                textAlign: 'center',
                color: 'rgba(255, 255, 255, 0.7)',
                wordBreak: 'break-word',
              }}
            >
              {asset.search_query || 'Educational Graphic'}
            </div>
          </div>
        )}
      </div>
    );
  }

  // Fallback for Lottie/Other types
  return (
    <div style={layoutStyle}>
      <div
        style={{
          width: '100%',
          height: '100%',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          background: 'rgba(255, 255, 255, 0.02)',
          borderRadius: '12px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <span style={{ fontSize: '12px', opacity: 0.5 }}>{asset.type}</span>
      </div>
    </div>
  );
};
