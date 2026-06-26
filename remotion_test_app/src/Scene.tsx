import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, Easing, Audio, staticFile } from 'remotion';
import { Scene, Asset, Connection } from './types';
import { AssetView } from './Asset';
import { ConnectionsView } from './Connections';

// Import template components
import { TitleSlide } from './templates/TitleSlide';
import { ConceptDiagram } from './templates/ConceptDiagram';
import { HorizontalTimeline } from './templates/HorizontalTimeline';
import { ColumnComparison } from './templates/ColumnComparison';
import { DatabaseGrid } from './templates/DatabaseGrid';
import { DynamicIllustration } from './templates/DynamicIllustration';

interface SceneViewProps {
  scene: Scene;
  theme: 'indigo' | 'gold' | 'emerald' | 'rose';
  layoutMode: string;
  globalConnections: Connection[];
  globalAssets: Asset[];
}

export const SceneView: React.FC<SceneViewProps> = ({
  scene,
  theme,
  layoutMode,
  globalConnections = [],
  globalAssets = [],
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const localAssets = scene.local_assets || scene.assets || [];
  const allAssets = [...globalAssets, ...localAssets];

  // Camera zoom and pan interpolation
  const camera = scene.camera || { focus_x: 50, focus_y: 50, zoom: 1.0, transition_duration: 1.0 };
  const transitionFrames = Math.max(1, (camera.transition_duration || 1.0) * fps);

  const zoom = interpolate(
    frame,
    [0, transitionFrames],
    [1.0, camera.zoom || 1.0],
    {
      extrapolateRight: 'clamp',
      easing: Easing.bezier(0.25, 0.1, 0.25, 1.0),
    }
  );

  const focusX = interpolate(
    frame,
    [0, transitionFrames],
    [50, camera.focus_x ?? 50],
    {
      extrapolateRight: 'clamp',
      easing: Easing.bezier(0.25, 0.1, 0.25, 1.0),
    }
  );

  const focusY = interpolate(
    frame,
    [0, transitionFrames],
    [50, camera.focus_y ?? 50],
    {
      extrapolateRight: 'clamp',
      easing: Easing.bezier(0.25, 0.1, 0.25, 1.0),
    }
  );

  // SVG connectors local to this scene + global connections
  const sceneConnections = globalConnections; // apply all connections

  // Let's format the audio URL
  let resolvedAudioUrl = '';
  if (scene.audio_url) {
    // Check if it starts with slash
    resolvedAudioUrl = scene.audio_url.startsWith('/') ? scene.audio_url : `/${scene.audio_url}`;
  }

  // --- RENDER ROUTER: TEMPLATE VS LEGACY ---
  if (scene.template_id) {
    const data = scene.template_data || {};
    
    return (
      <div style={{ width: '100%', height: '100%', position: 'relative' }}>
        {/* Background narration audio */}
        {resolvedAudioUrl && (
          <Audio src={staticFile(resolvedAudioUrl)} />
        )}

        {/* Selected Template Component Rendering */}
        {scene.template_id === 'title_slide' && (
          <TitleSlide
            title={data.title || ''}
            subtitle={data.subtitle || ''}
            icon_name={data.icon_name || 'book-open'}
            theme={theme}
          />
        )}

        {scene.template_id === 'concept_diagram' && (
          <ConceptDiagram
            left_title={data.left_title || ''}
            left_bullets={data.left_bullets || []}
            central_node={data.central_node || ''}
            leaf_nodes={data.leaf_nodes || []}
            theme={theme}
          />
        )}

        {scene.template_id === 'horizontal_timeline' && (
          <HorizontalTimeline
            timeline_title={data.timeline_title || ''}
            stages={data.stages || []}
            theme={theme}
          />
        )}

        {scene.template_id === 'column_comparison' && (
          <ColumnComparison
            left_column={data.left_column || { header: '', bullets: [] }}
            right_column={data.right_column || { header: '', bullets: [] }}
            theme={theme}
          />
        )}

        {scene.template_id === 'database_grid' && (
          <DatabaseGrid
            table_title={data.table_title || ''}
            headers={data.headers || []}
            rows={data.rows || []}
            highlight_row_idx={data.highlight_row_idx}
            highlight_col_idx={data.highlight_col_idx}
            theme={theme}
          />
        )}

        {scene.template_id === 'illustrated_scene' && (
          <DynamicIllustration
            title={data.title || ''}
            svg_elements={data.svg_elements || []}
            animation_action={data.animation_action || 'none'}
            theme={theme}
            canvas_color={data.canvas_color}
          />
        )}
      </div>
    );
  }

  // Fallback to legacy asset positioning
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Background narration audio */}
      {resolvedAudioUrl && (
        <Audio src={staticFile(resolvedAudioUrl)} />
      )}

      {/* Virtual Camera Wrapper */}
      <div
        style={{
          width: '100%',
          height: '100%',
          position: 'absolute',
          transform: `scale(${zoom}) translate(${50 - focusX}%, ${50 - focusY}%)`,
          transformOrigin: '50% 50%',
          transition: 'transform 0.05s linear',
        }}
      >
        {/* Connection arrows / lines in background */}
        <ConnectionsView connections={sceneConnections} assets={allAssets} theme={theme} />

        {/* Global and Local Assets */}
        {allAssets.map((asset) => (
          <AssetView key={asset.id} asset={asset} theme={theme} />
        ))}
      </div>

      {/* Teacher Narration Subtitles Card */}
      {scene.teacher_script && (
        <div
          style={{
            position: 'absolute',
            bottom: 40,
            left: '10%',
            right: '10%',
            padding: '18px 28px',
            background: 'rgba(10, 15, 30, 0.7)',
            backdropFilter: 'blur(16px)',
            borderRadius: '20px',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            textAlign: 'center',
            fontSize: '18px',
            lineHeight: '1.6',
            color: '#e2e8f0',
            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.4)',
            zIndex: 90,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '12px',
          }}
        >
          <div
            style={{
              fontSize: '24px',
              opacity: 0.8,
              alignSelf: 'flex-start',
              marginTop: '-4px',
            }}
          >
            💬
          </div>
          <div style={{ fontWeight: 500, letterSpacing: '-0.2px' }}>
            {scene.teacher_script}
          </div>
        </div>
      )}
    </div>
  );
};
