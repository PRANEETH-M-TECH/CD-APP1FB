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

export interface Scene {
  scene_no: number;
  clip_no?: number;
  camera?: Camera;
  teacher_script: string;
  audio_url?: string;
  local_assets?: Asset[];
  assets?: Asset[];
  durationInFrames?: number; // populated dynamically
  template_id?: string;
  template_data?: any;
}

export interface Storyboard {
  lesson_title: string;
  layout_mode: 'timeline' | 'process' | 'comparison' | 'radial_breakdown';
  theme: 'indigo' | 'gold' | 'emerald' | 'rose';
  global_assets?: Asset[];
  connections?: Connection[];
  lesson_id: string;
  scenes: Scene[];
}
