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
import { ImageScene } from './templates/ImageScene';
import { ProcessImageScene } from './templates/ProcessImageScene';
import { CycleTemplate } from './templates/CycleTemplate';
import { MathDerivation } from './templates/MathDerivation';
import { VennDiagram } from './templates/VennDiagram';
import { TaxonomyTree } from './templates/TaxonomyTree';
import { CartesianGrid } from './templates/CartesianGrid';
import { GeoMarker } from './templates/GeoMarker';
import { BeforeAfterSlider } from './templates/BeforeAfterSlider';
import { QuizCheckpoint } from './templates/QuizCheckpoint';

interface SceneViewProps {
  scene: Scene;
  theme: string;
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

  // --- RENDER ROUTER: SEQUENTIAL PROCESS VS TEMPLATE VS LEGACY ---
  if (scene.visual_steps && scene.visual_steps.length > 0) {
    return (
      <div style={{ width: '100%', height: '100%', position: 'relative' }}>
        {/* Background narration audio */}
        {resolvedAudioUrl && (
          <Audio src={staticFile(resolvedAudioUrl)} />
        )}

        <ProcessImageScene
          visual_steps={scene.visual_steps}
          theme={theme as any}
          durationInFrames={scene.durationInFrames}
        />

        {/* Teacher Narration Subtitles */}
        {scene.teacher_script && (
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
            {scene.teacher_script}
          </div>
        )}
      </div>
    );
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

        {scene.template_id === 'cycle_template' && (
          <CycleTemplate
            title={data.title || ''}
            stages={data.stages || []}
            theme={theme}
          />
        )}

        {scene.template_id === 'math_derivation' && (
          <MathDerivation
            title={data.title || ''}
            formula={data.formula || ''}
            steps={data.steps || []}
            theme={theme}
          />
        )}

        {scene.template_id === 'venn_diagram' && (
          <VennDiagram
            left={data.left || []}
            right={data.right || []}
            intersection={data.intersection || []}
            left_title={data.left_title}
            right_title={data.right_title}
            theme={theme}
          />
        )}

        {scene.template_id === 'taxonomy_tree' && (
          <TaxonomyTree
            title={data.title || ''}
            root_label={data.root_label || ''}
            branches={data.branches || []}
            theme={theme}
          />
        )}

        {scene.template_id === 'cartesian_grid' && (
          <CartesianGrid
            title={data.title || ''}
            points={data.points || []}
            lines={data.lines || []}
            equation_label={data.equation_label}
            theme={theme}
            svg_elements={data.svg_elements || []}
          />
        )}

        {scene.template_id === 'horizontal_timeline' && (
          <HorizontalTimeline
            timeline_title={data.timeline_title || ''}
            stages={data.stages || []}
            theme={theme as any}
          />
        )}

        {scene.template_id === 'column_comparison' && (
          <ColumnComparison
            left_column={data.left_column || { header: '', bullets: [] }}
            right_column={data.right_column || { header: '', bullets: [] }}
            theme={theme}
          />
        )}

        {scene.template_id === 'geo_marker' && (
          <GeoMarker
            title={data.title || ''}
            map_name={data.map_name}
            markers={data.markers || []}
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

        {scene.template_id === 'before_after_slider' && (
          <BeforeAfterSlider
            title={data.title || ''}
            before_label={data.before_label}
            after_label={data.after_label}
            before_text={data.before_text}
            after_text={data.after_text}
            theme={theme}
          />
        )}

        {scene.template_id === 'quiz_checkpoint' && (
          <QuizCheckpoint
            question={data.question}
            options={data.options}
            correct_idx={data.correct_idx}
            theme={theme}
            left_title={data.left_title}
            left_bullets={data.left_bullets}
            right_column={data.right_column}
          />
        )}

        {scene.template_id === 'illustrated_scene' && (
          <DynamicIllustration
            title={data.title || ''}
            svg_elements={data.svg_elements || []}
            animation_action={data.animation_action || 'none'}
            theme={theme as any}
            canvas_color={data.canvas_color}
          />
        )}

        {scene.template_id === 'image_scene' && (
          <ImageScene
            title={data.title || ''}
            teacher_script={scene.teacher_script || ''}
            image_url={data.image_url || ''}
            zoom_targets={data.zoom_targets || []}
            annotations={data.annotations || []}
            motion_path={data.motion_path}
            spotlight={data.spotlight}
            animation_style={data.animation_style || 'simple_zoom'}
            theme={theme as any}
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
        <ConnectionsView connections={sceneConnections} assets={allAssets} theme={theme as any} />

        {/* Global and Local Assets */}
        {allAssets.map((asset) => (
          <AssetView key={asset.id} asset={asset} theme={theme as any} />
        ))}
      </div>

      {/* Teacher Narration Subtitles */}
      {scene.teacher_script && (
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
          {scene.teacher_script}
        </div>
      )}
    </div>
  );
};
