export interface Layout {
  top: number;
  left: number;
  width: number;
  height: number;
}

export interface Animation {
  type: 'fade_in' | 'fade_out' | 'slide_in_left' | 'slide_in_right' | 'scale_up' | 'scale_down' | 'spin' | 'appear' | 'disappear';
  start_time: number; // in seconds
  duration: number; // in seconds
}

export interface Asset {
  id: string;
  type: 'image' | 'icon' | 'text' | 'lottie';
  search_query?: string;
  asset_url?: string;
  text_content?: string;
  layout: Layout;
  animations?: Animation[];
}

export interface Connection {
  from: string;
  to: string;
  type: 'arrow' | 'line';
  color?: string;
}

export interface Camera {
  focus_x: number;
  focus_y: number;
  zoom: number;
  transition_duration: number; // in seconds
}

export interface SvgElement {
  type: 'circle' | 'rect' | 'ellipse' | 'line' | 'path';
  // Circle attributes
  cx?: number;
  cy?: number;
  r?: number;
  // Rect attributes
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  rx?: number;
  // Ellipse attributes (cx, cy shared with circle, rx shared with rect)
  ry?: number;
  // Line attributes
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
  // Path attributes
  d?: string;
  // Common styling
  fill?: string;
  stroke?: string;
  stroke_width?: number;
  // Animation flag
  animate?: boolean; // whether this element moves during action phase
  label?: string; // optional text label
}

export interface DynamicIllustrationData {
  title: string;
  svg_elements: SvgElement[];
  animation_action: 'rise' | 'fall' | 'spin' | 'scale_up' | 'slide_left' | 'slide_right' | 'none';
  canvas_color?: string; // optional background accent
}

export interface StepAnimation {
  transition: 'fade' | 'slide' | 'wipe' | 'none';
  camera_motion: 'zoom_in' | 'zoom_out' | 'pan_left' | 'pan_right' | 'none';
}

export interface StepContent {
  svg_elements?: SvgElement[];
  text_content?: string;
}

export interface VisualStep {
  step_no: number;
  visual_type: 'diagram' | 'equation' | 'table';
  focus?: string;
  duration_seconds: number;
  content: StepContent;
  animation: StepAnimation;
}

export interface Scene {
  scene_no: number;
  clip_no?: number;
  camera?: Camera;
  purpose: string;
  visual_strategy: string;
  teacher_script: string;
  audio_url?: string;
  local_assets?: Asset[];
  assets?: Asset[];
  durationInFrames?: number; // populated dynamically
  template_id?: 'title_slide' | 'concept_diagram' | 'cycle_template' | 'math_derivation' | 'venn_diagram' | 'taxonomy_tree' | 'cartesian_grid' | 'column_comparison' | 'geo_marker' | 'database_grid' | 'before_after_slider' | 'quiz_checkpoint' | 'horizontal_timeline' | 'illustrated_scene' | 'image_scene';
  visual_steps?: VisualStep[];
  template_data?: any;
}

export interface Storyboard {
  lesson_title: string;
  layout_mode: 'timeline' | 'process' | 'comparison' | 'radial_breakdown';
  theme: 'indigo' | 'gold' | 'emerald' | 'rose' | 'Science' | 'Math' | 'History' | 'Civics' | 'General';
  global_assets?: Asset[];
  connections?: Connection[];
  lesson_id: string;
  scenes: Scene[];
}

export interface ZoomTarget {
  x: number;           // % horizontal position (0-100)
  y: number;           // % vertical position (0-100)
  scale: number;       // zoom level (1 = normal, 2.5 = zoomed in)
  at_percent: number;  // when in scene timeline (0% = start, 100% = end)
}

export interface ImageAnnotation {
  type: 'arrow' | 'circle' | 'label';
  x: number;           // % position
  y: number;           // % position
  target_x?: number;   // arrow endpoint (for arrows only)
  target_y?: number;
  label?: string;      // text content
  color?: string;      // hex color
  at_percent: number;  // when to appear (0-100%)
}

export interface MotionPath {
  path_data: string;      // SVG path d attribute
  dot_color?: string;     // dot fill color
  dot_size?: number;      // dot radius
  start_percent?: number; // when motion starts (0-100%)
  duration_percent?: number; // how long motion takes (% of scene)
}

export interface SpotlightRegion {
  x: number;           // % center position
  y: number;
  radius: number;      // spotlight radius in px
  at_percent: number;  // when to activate
}

export interface ImageSceneData {
  title: string;
  image_prompt: string;
  image_url: string;
  zoom_targets?: ZoomTarget[];
  annotations?: ImageAnnotation[];
  motion_path?: MotionPath;
  spotlight?: SpotlightRegion;
  animation_style: 'zoom_and_annotate' | 'spotlight_reveal' | 'motion_path' | 'progressive_reveal' | 'simple_zoom';
}

