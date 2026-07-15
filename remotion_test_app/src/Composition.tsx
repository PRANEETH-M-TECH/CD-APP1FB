import React from 'react';
import { Sequence, AbsoluteFill } from 'remotion';
import { Storyboard, Scene } from './types';
import { SceneView } from './Scene';

import { getTheme } from './themeHelper';

export const ConveyorComposition: React.FC<Storyboard> = ({
  lesson_title,
  layout_mode,
  theme = 'indigo',
  global_assets = [],
  connections = [],
  scenes = [],
}) => {
  const activeTheme = getTheme(theme);
  const gradient = activeTheme.background;
  const accentColor = activeTheme.accentColor;

  let currentFrameOffset = 0;

  return (
    <AbsoluteFill
      style={{
        background: gradient,
        fontFamily: activeTheme.fontFamily,
        color: activeTheme.textColor,
        overflow: 'hidden',
      }}
    >
      {/* Header bar overlay (clean, no-blur floating text) */}
      <div
        style={{
          position: 'absolute',
          top: 30,
          left: 40,
          right: 40,
          height: 60,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '0 24px',
          zIndex: 100,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              backgroundColor: accentColor,
              boxShadow: `0 0 12px ${accentColor}`,
            }}
          />
          <h1
            style={{
              fontSize: '20px',
              fontWeight: 700,
              margin: 0,
              letterSpacing: '-0.5px',
              textShadow: '0 2px 4px rgba(0, 0, 0, 0.5)',
            }}
          >
            {lesson_title}
          </h1>
        </div>
        <div
          style={{
            fontSize: '12px',
            textTransform: 'uppercase',
            letterSpacing: '1.5px',
            fontWeight: 600,
            color: 'rgba(255, 255, 255, 0.5)',
            background: 'rgba(255, 255, 255, 0.05)',
            padding: '6px 12px',
            borderRadius: '20px',
            border: '1px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          {layout_mode} Mode
        </div>
      </div>

      {/* Render each scene in its own sequence */}
      {scenes.map((scene, idx) => {
        const startFrame = currentFrameOffset;
        const duration = scene.durationInFrames || 180;
        currentFrameOffset += duration;

        return (
          <Sequence
            key={`scene-${scene.scene_no}-${idx}`}
            from={startFrame}
            durationInFrames={duration}
          >
            <SceneView
              scene={scene}
              theme={theme}
              layoutMode={layout_mode}
              globalConnections={connections}
              globalAssets={global_assets}
            />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
