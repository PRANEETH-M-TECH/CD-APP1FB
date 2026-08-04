const LessonSceneGraph = require('../scene/LessonSceneGraph');
const Scene = require('../scene/Scene');
const SceneNode = require('../scene/SceneNode');
const ComponentFactory = require('../components/factory/ComponentFactory');

/**
 * StoryboardAdapter.js
 * Adapter layer responsible for converting storyboard JSONs between evolved schemas
 * and legacy layouts to guarantee backward compatibility, now storing Component models.
 */
class StoryboardAdapter {
  /**
   * Helper function to instantiate a SceneNode containing a component.
   */
  static createNode(nodeId, nodeType, compType, compId, properties = {}, style = {}, children = [], metadata = {}, visibility = true) {
    const compChildren = children.map(c => c.component).filter(Boolean);
    const component = ComponentFactory.create(compType, compId, properties, style, compChildren, metadata, visibility);
    return new SceneNode(nodeId, nodeType, component, children);
  }

  /**
   * Extracts { text, icon } from an LLM-provided item, whether it's a plain
   * string (no icon) or an object with a text-like field plus an optional
   * icon name (matched against hyperframes_engine/shared/icons.js at render
   * time, with an automatic fallback to a generic dot icon for unknown/
   * missing names - so a bad icon name never breaks rendering).
   */
  static extractTextAndIcon(item, extraTextKeys = []) {
    if (item == null) return { text: '', icon: null };
    if (typeof item !== 'object') return { text: String(item), icon: null };
    const textKeys = ['text', 'label', 'name', ...extraTextKeys];
    let text = '';
    for (const k of textKeys) {
      if (item[k]) { text = item[k]; break; }
    }
    const icon = item.icon || item.icon_name || item.iconName || null;
    return { text: typeof text === 'string' ? text : String(text || ''), icon };
  }

  /**
   * Adapts a storyboard JSON object into a unified LessonSceneGraph.
   * If the JSON uses legacy templates, it maps them into clean intermediate SceneNodes and Components.
   * @param {object} rawJson 
   * @returns {LessonSceneGraph}
   */
  static toSceneGraph(rawJson) {
    if (!rawJson) return null;

    // Adapt legacy storyboard JSONs to support the evolved planning metadata
    const LegacyPlannerAdapter = require('../planner/adapters/LegacyPlannerAdapter');
    const adaptedJson = LegacyPlannerAdapter.adapt(rawJson);

    // Validate the adapted storyboard for educational coverage
    const StoryboardValidator = require('../planner/validators/StoryboardValidator');
    const validation = StoryboardValidator.validate(adaptedJson);
    if (!validation.isValid) {
      console.warn(`[Storyboard Validation Warning] ${validation.errors.join('; ')}`);
    }

    const TeachingModel = require('../teaching/models/TeachingModel');
    const TeachingStep = require('../teaching/models/TeachingStep');

    const title = adaptedJson.lesson_title || adaptedJson.title || 'Untitled Lesson';
    const theme = adaptedJson.theme || 'indigo';
    const layoutMode = adaptedJson.layout_mode || 'process';
    const metadata = adaptedJson.metadata || {};

    const scenes = (adaptedJson.scenes || []).map((sceneJson) => {
      // 1. If the scene already has evolved scene graph nodes, deserialize directly
      if (sceneJson.nodes && sceneJson.nodes.length > 0) {
        return Scene.deserialize(sceneJson);
      }

      // 2. Otherwise, adapt legacy fields to evolved Scene Graph nodes containing Components
      const sceneNo = sceneJson.scene_no;
      // Normalize template id: accept CamelCase, spaces, and hyphens from LLM outputs
      const rawTemplateId = sceneJson.template_id || 'general_scene';
      const normalizeTemplateId = (t) => {
        if (!t) return 'general_scene';
        // If already snake_case, return lowercased
        let s = String(t).trim();
        // Convert CamelCase to snake_case, and replace spaces/hyphens with underscore
        s = s.replace(/([a-z0-9])([A-Z])/g, '$1_$2');
        s = s.replace(/[^a-zA-Z0-9]+/g, '_');
        s = s.toLowerCase();
        s = s.replace(/__+/g, '_').replace(/^_+|_+$/g, '');
        if (!s) return 'general_scene';
        return s;
      };
      const templateId = normalizeTemplateId(rawTemplateId || 'general_scene');
      const teacherScript = sceneJson.teacher_script || '';
      const data = sceneJson.template_data || {};

      // Parse metadata
      const sceneMetadata = {
        title: data.title || data.timeline_title || data.table_title || '',
        description: data.description || '',
        ...sceneJson.metadata
      };

      const LegacyAssetAdapter = require('../assets/adapters/LegacyAssetAdapter');
      const resolvedAudio = LegacyAssetAdapter.resolveFilename(sceneJson.audio_url, { subject: rawJson.subject });

      // Parse timeline configuration
      const timeline = {
        animation_style: data.animation_style || 'stagger_in',
        animation_action: data.animation_action || 'none',
        audio_url: resolvedAudio ? resolvedAudio.path : (sceneJson.audio_url || null),
        audio_asset: resolvedAudio ? resolvedAudio.serialize() : null,
        durationInFrames: sceneJson.durationInFrames || null,
        ...sceneJson.timeline
      };

      const nodes = [];

      // Map template data to structured data nodes wrapping component systems
      switch (templateId) {
        case 'title_slide': {
          nodes.push(StoryboardAdapter.createNode(`title_${sceneNo}`, 'TEXT', 'TEXT', `title_comp_${sceneNo}`, { text: data.title || '' }));
          nodes.push(StoryboardAdapter.createNode(`subtitle_${sceneNo}`, 'TEXT', 'TEXT', `subtitle_comp_${sceneNo}`, { text: data.subtitle || '' }));
          nodes.push(StoryboardAdapter.createNode(`icon_card_${sceneNo}`, 'CUSTOM', 'CUSTOM', `icon_comp_${sceneNo}`, { type: 'icon', name: 'shield' }));
          break;
        }

        case 'concept_diagram': {
          // Normalize LLM aliases → central_node + leaf_nodes
          const centerRaw = data.central_node
            || data.central_concept
            || data.main_concept
            || data.concept_name
            || data.title
            || data.question
            || data.header
            || '';
          const centerIcon = (centerRaw && typeof centerRaw === 'object')
            ? (centerRaw.icon || centerRaw.icon_name || null)
            : (data.central_icon || data.icon || null);
          data.central_node = typeof centerRaw === 'object'
            ? (centerRaw.text || centerRaw.label || centerRaw.name || '')
            : String(centerRaw || '');

          let rawLeaves = (data.leaf_nodes && Array.isArray(data.leaf_nodes))
            ? data.leaf_nodes
            : (data.key_nodes || data.branches || data.attributes || data.nodes || data.options || data.stages || data.steps || data.bullets || data.items || data.concepts || data.elements || []);

          if (!Array.isArray(rawLeaves) && typeof rawLeaves === 'object' && rawLeaves !== null) {
            rawLeaves = Object.entries(rawLeaves).map(([key, val]) => `${key}: ${val}`);
          }
          if (!Array.isArray(rawLeaves)) {
            rawLeaves = [];
          }

          // Icons captured per-leaf (aligned by index) before leaf_nodes gets
          // flattened to plain display strings below, which drop object shape.
          const leafIcons = rawLeaves.map((leaf) =>
            (leaf && typeof leaf === 'object') ? (leaf.icon || leaf.icon_name || null) : null
          );

          data.leaf_nodes = rawLeaves.map((leaf) => {
            if (typeof leaf === 'string') return leaf;
            if (!leaf || typeof leaf !== 'object') return String(leaf || '');
            const label = leaf.text || leaf.label || leaf.name || '';
            const detail = leaf.value || leaf.description || '';
            if (label && detail) return `${label}: ${detail}`;
            if (Array.isArray(leaf.sub_branches)) {
              return label ? `${label} (${leaf.sub_branches.join(', ')})` : leaf.sub_branches.join(', ');
            }
            return label || detail || '';
          }).filter(Boolean);

          nodes.push(StoryboardAdapter.createNode(`center_${sceneNo}`, 'TEXT', 'TEXT', `center_comp_${sceneNo}`, { text: data.central_node, icon: centerIcon }));

          const leafNodes = data.leaf_nodes.map((leaf, idx) =>
            StoryboardAdapter.createNode(`leaf_${sceneNo}_${idx}`, 'TEXT', 'TEXT', `leaf_comp_${sceneNo}_${idx}`, { text: leaf, icon: leafIcons[idx] || null })
          );
          nodes.push(StoryboardAdapter.createNode(`leaves_${sceneNo}`, 'GROUP', 'GROUP', `leaves_comp_${sceneNo}`, {}, {}, leafNodes));

          if (data.left_bullets && data.left_bullets.length > 0) {
            nodes.push(StoryboardAdapter.createNode(`left_title_${sceneNo}`, 'TEXT', 'TEXT', `left_title_comp_${sceneNo}`, { text: data.left_title || '' }));
            const bulletNodes = data.left_bullets.map((bullet, idx) =>
              StoryboardAdapter.createNode(`bullet_${sceneNo}_${idx}`, 'TEXT', 'TEXT', `bullet_comp_${sceneNo}_${idx}`, { text: bullet })
            );
            nodes.push(StoryboardAdapter.createNode(`bullets_${sceneNo}`, 'GROUP', 'GROUP', `bullets_comp_${sceneNo}`, {}, {}, bulletNodes));
          }
          break;
        }

        case 'cycle_template': {
          nodes.push(StoryboardAdapter.createNode(`title_${sceneNo}`, 'TEXT', 'TEXT', `title_comp_${sceneNo}`, { text: data.title || '' }));

          // LLM: cycle_elements[{id,label,description}] → stages
          if (!(data.stages && data.stages.length) && Array.isArray(data.cycle_elements)) {
            data.stages = data.cycle_elements.map((el) => {
              if (typeof el === 'string') return el;
              return el.label || el.name || el.text || el.id || '';
            }).filter(Boolean);
          }

          const stageNodes = (data.stages || []).map((stage, idx) => {
            const label = typeof stage === 'object' ? (stage.label || stage.name || stage.text || '') : stage;
            const icon = typeof stage === 'object' ? (stage.icon || stage.icon_name || null) : null;
            return StoryboardAdapter.createNode(`stage_${sceneNo}_${idx}`, 'TEXT', 'TEXT', `stage_comp_${sceneNo}_${idx}`, { text: label, icon });
          });
          nodes.push(StoryboardAdapter.createNode(`stages_${sceneNo}`, 'GROUP', 'GROUP', `stages_comp_${sceneNo}`, {}, {}, stageNodes));
          nodes.push(StoryboardAdapter.createNode(`orbit_dot_${sceneNo}`, 'SHAPE', 'SHAPE', `orbit_dot_comp_${sceneNo}`, { shapeType: 'circle' }));
          break;
        }

        case 'math_derivation': {
          nodes.push(StoryboardAdapter.createNode(`title_${sceneNo}`, 'TEXT', 'TEXT', `title_comp_${sceneNo}`, { text: data.title || '' }));
          if (data.formula) {
            nodes.push(StoryboardAdapter.createNode(`formula_${sceneNo}`, 'CUSTOM', 'CUSTOM', `formula_comp_${sceneNo}`, { latex: data.formula }));
          }

          // LLM may send `steps` (string[]) or `equation_steps` ({step, value}[])
          const rawSteps = (data.steps && data.steps.length)
            ? data.steps
            : (data.equation_steps || []);
          const normalizedSteps = rawSteps.map((step) => {
            if (typeof step === 'string') return step;
            if (step && typeof step === 'object') {
              const label = step.step || step.label || step.name || '';
              const value = step.value || step.result || step.expression || '';
              if (label && value) return `${label}: ${value}`;
              return label || value || '';
            }
            return String(step || '');
          }).filter(Boolean);

          if (data.final_answer_label || data.final_answer_value) {
            const label = data.final_answer_label || 'Answer';
            const value = data.final_answer_value || '';
            normalizedSteps.push(`${label} ${value}`.trim());
          }

          // Keep template_data.steps in sync for KaTeX/DOM population
          data.steps = normalizedSteps;

          const stepNodes = normalizedSteps.map((step, idx) =>
            StoryboardAdapter.createNode(`step_${sceneNo}_${idx}`, 'TEXT', 'TEXT', `step_comp_${sceneNo}_${idx}`, { text: step })
          );
          nodes.push(StoryboardAdapter.createNode(`steps_${sceneNo}`, 'GROUP', 'GROUP', `steps_comp_${sceneNo}`, {}, {}, stepNodes));
          break;
        }

        case 'column_comparison': {
          nodes.push(StoryboardAdapter.createNode(`title_${sceneNo}`, 'TEXT', 'TEXT', `title_comp_${sceneNo}`, { text: data.title || '' }));

          let leftCol = data.left_column || data.left_col;
          let rightCol = data.right_column || data.right_col;

          // Shape: columns[{title, items:[{text}|string]}]
          if (!leftCol && !rightCol && Array.isArray(data.columns) && data.columns.length >= 2) {
            const toBullets = (col) => (col.items || col.bullets || []).map((item) => {
              if (typeof item === 'string') return item;
              return item.text || item.label || item.value || '';
            }).filter(Boolean);
            leftCol = { header: data.columns[0].title || data.columns[0].header || 'A', bullets: toBullets(data.columns[0]) };
            rightCol = { header: data.columns[1].title || data.columns[1].header || 'B', bullets: toBullets(data.columns[1]) };
          }

          // Shape: column_titles/column_headers + rows[{label,value1,value2}] or row dicts keyed by header
          if (!leftCol && !rightCol && (data.column_titles || data.column_headers || data.rows)) {
            const titles = data.column_titles || data.column_headers || ['A', 'B'];
            const rows = data.rows || [];
            const leftKey = titles[0];
            const rightKey = titles[1];
            leftCol = {
              header: leftKey || 'A',
              bullets: rows.map((r) => {
                if (typeof r === 'string') return r;
                if (Array.isArray(r)) return String(r[0] != null ? r[0] : '');
                const label = r.label || r.name || '';
                const value = r.value1 != null ? r.value1
                  : (r.left != null ? r.left
                    : (leftKey && r[leftKey] != null ? r[leftKey] : ''));
                return label ? `${label}: ${value}` : String(value);
              }).filter((b) => b !== '')
            };
            rightCol = {
              header: rightKey || 'B',
              bullets: rows.map((r) => {
                if (typeof r === 'string') return r;
                if (Array.isArray(r)) return String(r[1] != null ? r[1] : '');
                const label = r.label || r.name || '';
                const value = r.value2 != null ? r.value2
                  : (r.right != null ? r.right
                    : (rightKey && r[rightKey] != null ? r[rightKey] : ''));
                return label ? `${label}: ${value}` : String(value);
              }).filter((b) => b !== '')
            };
          }
          if (!leftCol && data.left) {
            leftCol = { header: data.left_title || 'A', bullets: data.left };
          }
          if (!rightCol && data.right) {
            rightCol = { header: data.right_title || 'B', bullets: data.right };
          }
          leftCol = leftCol || { header: '', bullets: [] };
          rightCol = rightCol || { header: '', bullets: [] };
          data.left_column = leftCol;
          data.right_column = rightCol;

          const leftBulletNodes = (leftCol.bullets || []).map((bullet, idx) => {
            const { text, icon } = StoryboardAdapter.extractTextAndIcon(bullet);
            return StoryboardAdapter.createNode(`left_bullet_${sceneNo}_${idx}`, 'TEXT', 'TEXT', `left_bullet_comp_${sceneNo}_${idx}`, { text, icon });
          });
          nodes.push(StoryboardAdapter.createNode(`left_col_${sceneNo}`, 'GROUP', 'GROUP', `left_col_comp_${sceneNo}`, { header: leftCol.header }, {}, leftBulletNodes));

          const rightBulletNodes = (rightCol.bullets || []).map((bullet, idx) => {
            const { text, icon } = StoryboardAdapter.extractTextAndIcon(bullet);
            return StoryboardAdapter.createNode(`right_bullet_${sceneNo}_${idx}`, 'TEXT', 'TEXT', `right_bullet_comp_${sceneNo}_${idx}`, { text, icon });
          });
          nodes.push(StoryboardAdapter.createNode(`right_col_${sceneNo}`, 'GROUP', 'GROUP', `right_col_comp_${sceneNo}`, { header: rightCol.header }, {}, rightBulletNodes));
          break;
        }

        case 'horizontal_timeline': {
          // LLM: title + events[{title|label}] → timeline_title + stages[{label, step_no}]
          data.timeline_title = data.timeline_title || data.title || '';
          if (!(data.stages && data.stages.length) && Array.isArray(data.events)) {
            data.stages = data.events.map((ev, idx) => {
              if (typeof ev === 'string') return { label: ev, step_no: idx + 1 };
              return {
                label: ev.label || ev.title || ev.name || ev.description || `Step ${idx + 1}`,
                step_no: ev.step_no != null ? ev.step_no : (idx + 1)
              };
            });
          }

          nodes.push(StoryboardAdapter.createNode(`title_${sceneNo}`, 'TEXT', 'TEXT', `title_comp_${sceneNo}`, { text: data.timeline_title || '' }));
          const stageNodes = (data.stages || []).map((stage, idx) => {
            const label = typeof stage === 'object'
              ? (stage.label || stage.title || stage.name || '')
              : stage;
            const stepNo = typeof stage === 'object' && stage.step_no != null ? stage.step_no : (idx + 1);
            return StoryboardAdapter.createNode(
              `stage_${sceneNo}_${idx}`, 'TEXT', 'TEXT', `stage_comp_${sceneNo}_${idx}`,
              { text: label }, {}, [], { step_no: stepNo }
            );
          });
          nodes.push(StoryboardAdapter.createNode(`stages_${sceneNo}`, 'GROUP', 'GROUP', `stages_comp_${sceneNo}`, {}, {}, stageNodes));
          break;
        }

        case 'database_grid': {
          nodes.push(StoryboardAdapter.createNode(`title_${sceneNo}`, 'TEXT', 'TEXT', `title_comp_${sceneNo}`, { text: data.table_title || data.title || '' }));
          
          if (!data.headers && !data.rows && Array.isArray(data.items)) {
            const firstItem = data.items[0] || {};
            const keys = Object.keys(firstItem);
            data.headers = keys.map(k => k.charAt(0).toUpperCase() + k.slice(1));
            data.rows = data.items.map(item => keys.map(k => item[k] || ''));
          }

          const headNodes = (data.headers || []).map((h, idx) =>
            StoryboardAdapter.createNode(`header_${sceneNo}_${idx}`, 'TEXT', 'TEXT', `header_comp_${sceneNo}_${idx}`, { text: h })
          );
          nodes.push(StoryboardAdapter.createNode(`headers_${sceneNo}`, 'GROUP', 'GROUP', `headers_comp_${sceneNo}`, {}, {}, headNodes));

          const rowNodes = (data.rows || []).map((row, rIdx) => {
            const cells = Array.isArray(row) ? row : Object.values(row || {});
            const cellNodes = cells.map((cell, cIdx) =>
              StoryboardAdapter.createNode(`cell_${sceneNo}_${rIdx}_${cIdx}`, 'TEXT', 'TEXT', `cell_comp_${sceneNo}_${rIdx}_${cIdx}`, { text: cell })
            );
            return StoryboardAdapter.createNode(`row_${sceneNo}_${rIdx}`, 'GROUP', 'GROUP', `row_comp_${sceneNo}_${rIdx}`, {}, {}, cellNodes);
          });
          nodes.push(StoryboardAdapter.createNode(`rows_${sceneNo}`, 'GROUP', 'GROUP', `rows_comp_${sceneNo}`, {}, {}, rowNodes));
          break;
        }

        case 'venn_diagram': {
          // Normalize LLM keys & fallbacks
          let leftList = data.left || data.left_items || data.left_list || (data.left_col ? (data.left_col.bullets || data.left_col.items) : null) || [];
          let rightList = data.right || data.right_items || data.right_list || (data.right_col ? (data.right_col.bullets || data.right_col.items) : null) || [];
          let interList = data.intersection || data.intersect || data.shared || data.overlap || data.middle || data.intersection_items || [];
          let leftTitleVal = data.left_title || (data.left_col ? (data.left_col.header || data.left_col.title) : null) || 'A';
          let rightTitleVal = data.right_title || (data.right_col ? (data.right_col.header || data.right_col.title) : null) || 'B';

          data.left = leftList;
          data.right = rightList;
          data.intersection = interList;
          data.left_title = leftTitleVal;
          data.right_title = rightTitleVal;

          nodes.push(StoryboardAdapter.createNode(`left_title_${sceneNo}`, 'TEXT', 'TEXT', `left_title_comp_${sceneNo}`, { text: leftTitleVal }));
          nodes.push(StoryboardAdapter.createNode(`right_title_${sceneNo}`, 'TEXT', 'TEXT', `right_title_comp_${sceneNo}`, { text: rightTitleVal }));
          
          const leftNodes = leftList.map((item, idx) => {
            const { text, icon } = StoryboardAdapter.extractTextAndIcon(item);
            return StoryboardAdapter.createNode(`left_item_${sceneNo}_${idx}`, 'TEXT', 'TEXT', `left_item_comp_${sceneNo}_${idx}`, { text, icon });
          });
          nodes.push(StoryboardAdapter.createNode(`left_${sceneNo}`, 'GROUP', 'GROUP', `left_comp_${sceneNo}`, {}, {}, leftNodes));

          const midNodes = interList.map((item, idx) => {
            const { text, icon } = StoryboardAdapter.extractTextAndIcon(item);
            return StoryboardAdapter.createNode(`intersection_item_${sceneNo}_${idx}`, 'TEXT', 'TEXT', `intersection_item_comp_${sceneNo}_${idx}`, { text, icon });
          });
          nodes.push(StoryboardAdapter.createNode(`intersection_${sceneNo}`, 'GROUP', 'GROUP', `intersection_comp_${sceneNo}`, {}, {}, midNodes));

          const rightNodes = rightList.map((item, idx) => {
            const { text, icon } = StoryboardAdapter.extractTextAndIcon(item);
            return StoryboardAdapter.createNode(`right_item_${sceneNo}_${idx}`, 'TEXT', 'TEXT', `right_item_comp_${sceneNo}_${idx}`, { text, icon });
          });
          nodes.push(StoryboardAdapter.createNode(`right_${sceneNo}`, 'GROUP', 'GROUP', `right_comp_${sceneNo}`, {}, {}, rightNodes));
          break;
        }

        case 'quiz_checkpoint': {
          // Flatten multi-question wrapper: questions[0]
          if ((!data.question || !(data.options && data.options.length)) && Array.isArray(data.questions) && data.questions.length > 0) {
            const q0 = data.questions[0];
            data.question = data.question || q0.question || q0.text || '';
            data.options = data.options || q0.options || [];
            data.explanation = data.explanation || q0.explanation || '';
            data.correct_answer = data.correct_answer || q0.correct_answer;
            if (data.correct_answer_index == null) data.correct_answer_index = q0.correct_answer_index;
            if (data.correct_option_index == null) data.correct_option_index = q0.correct_option_index;
            data.correct_answer_id = data.correct_answer_id || q0.correct_answer_id;
          }

          if (data.correct_answer_index == null && data.correct_option_index != null) {
            data.correct_answer_index = data.correct_option_index;
          }

          // Normalize options and stamp is_correct for animate()
          const rawOptions = data.options || [];
          data.options = rawOptions.map((opt, idx) => {
            const text = typeof opt === 'object' ? (opt.text || opt.label || '') : String(opt);
            const id = typeof opt === 'object' ? opt.id : undefined;
            let isCorrect = false;
            if (typeof opt === 'object' && opt.is_correct != null) {
              isCorrect = !!opt.is_correct;
            } else if (data.correct_answer_id != null && id != null) {
              isCorrect = String(id) === String(data.correct_answer_id);
            } else if (data.correct_answer_index != null) {
              isCorrect = idx === data.correct_answer_index;
            } else if (data.correct_answer != null) {
              const ans = String(data.correct_answer).trim().toLowerCase();
              const t = text.trim().toLowerCase();
              isCorrect = t === ans || t.startsWith(ans) || ans.includes(t);
            }
            return { id, text, is_correct: isCorrect };
          });

          nodes.push(StoryboardAdapter.createNode(`question_${sceneNo}`, 'TEXT', 'TEXT', `question_comp_${sceneNo}`, { text: data.question || '' }));
          const optNodes = data.options.map((opt, idx) =>
            StoryboardAdapter.createNode(
              `option_${sceneNo}_${idx}`, 'TEXT', 'TEXT', `option_comp_${sceneNo}_${idx}`,
              { text: opt.text }, { isCorrect: !!opt.is_correct }
            )
          );
          nodes.push(StoryboardAdapter.createNode(`options_${sceneNo}`, 'GROUP', 'GROUP', `options_comp_${sceneNo}`, {}, {}, optNodes));
          if (data.explanation) {
            nodes.push(StoryboardAdapter.createNode(`explanation_${sceneNo}`, 'TEXT', 'TEXT', `explanation_comp_${sceneNo}`, { text: data.explanation }));
          }
          break;
        }

        case 'illustrated_scene': {
          nodes.push(StoryboardAdapter.createNode(`title_${sceneNo}`, 'TEXT', 'TEXT', `title_comp_${sceneNo}`, { text: data.title || '' }));
          
          const els = data.svg_elements || data.elements || [];
          const canvasChildNodes = els.map((el, idx) => {
            let elType = (el.type || 'rect').toLowerCase();

            // text_overlay → TEXT with flattened position coordinates
            if (elType === 'text_overlay' || elType === 'text') {
              const pos = el.position || {};
              const textProps = {
                text: el.text || el.label || '',
                x: el.x != null ? el.x : (pos.x != null ? pos.x : 0),
                y: el.y != null ? el.y : (pos.y != null ? pos.y : 0)
              };
              return StoryboardAdapter.createNode(
                `el_${sceneNo}_${idx}`, 'TEXT', 'TEXT', `el_comp_${sceneNo}_${idx}`,
                textProps, {}, []
              );
            }

            // Alias LLM drawing types onto supported SVG shapeTypes
            if (elType === 'dashed_line') elType = 'line';
            if (elType === 'animated_path') elType = 'path';

            const shapeType = ['path', 'circle', 'line', 'rect', 'ellipse'].includes(elType)
              ? elType
              : (el.path_data || el.d ? 'path' : 'rect');

            const stroke = el.stroke || el.stroke_color || '#ffffff';
            const fill = el.fill != null ? el.fill : 'none';
            const strokeWidth = el.stroke_width != null ? el.stroke_width : (el.strokeWidth != null ? el.strokeWidth : 2);
            const dashArray = el.dash_array || el.stroke_dasharray || el.strokeDasharray
              || ((el.type || '').toLowerCase() === 'dashed_line' ? '8 6' : '');

            const properties = {
              ...el,
              shapeType,
              stroke,
              fill,
              strokeWidth,
              dash_array: dashArray,
              d: el.d || el.path_data || '',
              path_data: el.path_data || el.d || ''
            };
            delete properties.type;
            delete properties.label;
            delete properties.stroke_color;
            delete properties.position;

            // Push fill/stroke into style so ComponentFactory defaults don't override presentation
            const shapeStyle = { fill, stroke, strokeWidth };

            const shapeChildren = [];
            if (el.label) {
              const labelPos = el.position || {};
              shapeChildren.push(StoryboardAdapter.createNode(
                `label_${sceneNo}_${idx}`, 'LABEL', 'LABEL', `label_comp_${sceneNo}_${idx}`,
                {
                  text: el.label,
                  targetId: `el_${sceneNo}_${idx}`,
                  x: labelPos.x != null ? labelPos.x : (el.cx || el.x || 0),
                  y: labelPos.y != null ? labelPos.y : (el.cy || el.y || 0)
                }
              ));
            }

            return StoryboardAdapter.createNode(
              `el_${sceneNo}_${idx}`, 'SHAPE', 'SHAPE', `el_comp_${sceneNo}_${idx}`,
              properties, shapeStyle, shapeChildren
            );
          });
          nodes.push(StoryboardAdapter.createNode(
            `canvas_${sceneNo}`, 'SVG', 'SVG', `canvas_comp_${sceneNo}`,
            { color: data.canvas_color || 'transparent', viewBox: '0 0 1280 720' },
            { width: '1280px', height: '720px', position: 'absolute', top: '0', left: '0' },
            canvasChildNodes
          ));
          break;
        }

        case 'image_scene': {
          const LegacyAssetAdapter = require('../assets/adapters/LegacyAssetAdapter');
          const resolvedImage = LegacyAssetAdapter.resolveFilename(data.image_url, { subject: rawJson.subject });
          nodes.push(StoryboardAdapter.createNode(
            `image_${sceneNo}`,
            'IMAGE',
            'IMAGE',
            `image_comp_${sceneNo}`,
            {
              url: resolvedImage ? resolvedImage.path : (data.image_url || ''),
              resolvedAsset: resolvedImage ? resolvedImage.serialize() : null
            }
          ));
          break;
        }

        case 'taxonomy_tree': {
          nodes.push(StoryboardAdapter.createNode(`title_${sceneNo}`, 'TEXT', 'TEXT', `title_comp_${sceneNo}`, { text: data.title || '' }));
          
          let rootLabel = 'Root Category';
          let rootIcon = null;
          let branches = [];

          if (data.root) {
            if (typeof data.root === 'object') {
              rootLabel = data.root.label || data.root.name || 'Root Category';
              rootIcon = data.root.icon || data.root.icon_name || null;
              branches = data.root.children || data.root.branches || [];
            } else {
              rootLabel = String(data.root);
            }
          }

          if (branches.length === 0 && Array.isArray(data.branches)) {
            branches = data.branches;
          }

          nodes.push(StoryboardAdapter.createNode(`root_${sceneNo}`, 'TEXT', 'TEXT', `root_comp_${sceneNo}`, { text: rootLabel, icon: rootIcon }));
          
          const branchNodes = branches.map((b, idx) => {
            const label = typeof b === 'object' ? (b.label || b.title || b.name || '') : b;
            const icon = typeof b === 'object' ? (b.icon || b.icon_name || null) : null;
            const sub = typeof b === 'object' ? (b.children || b.sub_nodes || b.sub || b.sub_branches || []) : [];
            const subTexts = sub.map(s => typeof s === 'object' ? (s.label || s.title || s.name || '') : s);
            return StoryboardAdapter.createNode(
              `branch_${sceneNo}_${idx}`, 'TEXT', 'TEXT', `branch_comp_${sceneNo}_${idx}`,
              { text: label, sub: subTexts, icon }
            );
          });
          nodes.push(StoryboardAdapter.createNode(`branches_${sceneNo}`, 'GROUP', 'GROUP', `branches_comp_${sceneNo}`, {}, {}, branchNodes));
          break;
        }

        case 'cartesian_grid': {
          nodes.push(StoryboardAdapter.createNode(`title_${sceneNo}`, 'TEXT', 'TEXT', `title_comp_${sceneNo}`, { text: data.title || '' }));
          nodes.push(StoryboardAdapter.createNode(`equation_${sceneNo}`, 'TEXT', 'TEXT', `equation_comp_${sceneNo}`, { text: data.equation_label || '' }));
          const pointNodes = (data.points || []).map((p, idx) =>
            StoryboardAdapter.createNode(`point_${sceneNo}_${idx}`, 'CUSTOM', 'CUSTOM', `point_comp_${sceneNo}_${idx}`, { x: p.x, y: p.y, label: p.label })
          );
          nodes.push(StoryboardAdapter.createNode(`points_${sceneNo}`, 'GROUP', 'GROUP', `points_comp_${sceneNo}`, {}, {}, pointNodes));

          // The LLM's own drawn curve (svg_elements[0], a path) - previously
          // parsed nowhere, so renderCartesianGrid always fell back to a
          // hardcoded generic curve regardless of what was actually asked for.
          const curveEl = (data.svg_elements || []).find((el) => (el.type || '').toLowerCase() === 'path' && (el.d || el.path_data));
          if (curveEl) {
            nodes.push(StoryboardAdapter.createNode(`curve_${sceneNo}`, 'CUSTOM', 'CUSTOM', `curve_comp_${sceneNo}`, {
              d: curveEl.d || curveEl.path_data,
              stroke: curveEl.stroke || curveEl.stroke_color || '#38bdf8',
              stroke_width: curveEl.stroke_width != null ? curveEl.stroke_width : 4
            }));
          }
          break;
        }

        case 'geo_marker': {
          nodes.push(StoryboardAdapter.createNode(`title_${sceneNo}`, 'TEXT', 'TEXT', `title_comp_${sceneNo}`, { text: data.title || '' }));
          const markerNodes = (data.markers || []).map((m, idx) =>
            StoryboardAdapter.createNode(`marker_${sceneNo}_${idx}`, 'CUSTOM', 'CUSTOM', `marker_comp_${sceneNo}_${idx}`, { label: m.label, x: m.x, y: m.y, description: m.description, icon: m.icon || m.icon_name || null })
          );
          nodes.push(StoryboardAdapter.createNode(`markers_${sceneNo}`, 'GROUP', 'GROUP', `markers_comp_${sceneNo}`, {}, {}, markerNodes));
          break;
        }

        case 'before_after_slider': {
          nodes.push(StoryboardAdapter.createNode(`title_${sceneNo}`, 'TEXT', 'TEXT', `title_comp_${sceneNo}`, { text: data.title || '' }));
          
          const beforeLabel = (data.before && data.before.label) || 'BEFORE';
          const beforeBullets = (data.before && data.before.bullets) || [];
          const beforeNodes = beforeBullets.map((b, idx) => {
            const { text, icon } = StoryboardAdapter.extractTextAndIcon(b);
            return StoryboardAdapter.createNode(`before_bullet_${sceneNo}_${idx}`, 'TEXT', 'TEXT', `before_bullet_comp_${sceneNo}_${idx}`, { text, icon });
          });
          nodes.push(StoryboardAdapter.createNode(`before_group_${sceneNo}`, 'GROUP', 'GROUP', `before_group_comp_${sceneNo}`, { label: beforeLabel }, {}, beforeNodes));

          const afterLabel = (data.after && data.after.label) || 'AFTER';
          const afterBullets = (data.after && data.after.bullets) || [];
          const afterNodes = afterBullets.map((b, idx) => {
            const { text, icon } = StoryboardAdapter.extractTextAndIcon(b);
            return StoryboardAdapter.createNode(`after_bullet_${sceneNo}_${idx}`, 'TEXT', 'TEXT', `after_bullet_comp_${sceneNo}_${idx}`, { text, icon });
          });
          nodes.push(StoryboardAdapter.createNode(`after_group_${sceneNo}`, 'GROUP', 'GROUP', `after_group_comp_${sceneNo}`, { label: afterLabel }, {}, afterNodes));
          break;
        }

        case 'general_scene':
        default: {
          nodes.push(StoryboardAdapter.createNode(`title_${sceneNo}`, 'TEXT', 'TEXT', `title_comp_${sceneNo}`, { text: data.title || '' }));
          break;
        }
      }

      const Camera = require('../camera/models/Camera');
      const camera = sceneJson.camera ? Camera.deserialize(sceneJson.camera) : new Camera();

      const Layout = require('../layout/models/Layout');
      const layout = sceneJson.layout ? Layout.deserialize(sceneJson.layout) : new Layout();

      const Focus = require('../focus/models/Focus');
      const focuses = (sceneJson.focuses || []).map(f => Focus.deserialize(f));

      const themeId = sceneJson.theme_id || 'default';

      // Build the adapted Scene object
      const adaptedScene = new Scene(
        sceneNo,
        templateId,
        teacherScript,
        sceneMetadata,
        nodes,
        timeline,
        camera,
        layout,
        focuses,
        themeId
      );

      // Preserve all other legacy layout properties so the compiler templates continue to read them exactly as before
      adaptedScene.template_data = data;
      adaptedScene.audio_url = sceneJson.audio_url;
      adaptedScene.durationInFrames = sceneJson.durationInFrames;

      return adaptedScene;
    });

    const PedagogicalStrategy = require('../pedagogy/models/PedagogicalStrategy');
    const StrategyRegistry = require('../pedagogy/registry/StrategyRegistry');
    const PedagogicalStrategyEngine = require('../pedagogy/engine/PedagogicalStrategyEngine');

    let pedagogy = null;
    if (adaptedJson.pedagogy) {
      pedagogy = PedagogicalStrategy.deserialize(adaptedJson.pedagogy);
    } else {
      // Load standard sequential strategy from library
      pedagogy = StrategyRegistry.getStrategy('default');
    }

    let teaching = null;
    if (adaptedJson.teaching) {
      teaching = TeachingModel.deserialize(adaptedJson.teaching);
    } else if (adaptedJson.teaching_plan) {
      teaching = TeachingModel.deserialize(adaptedJson.teaching_plan);
    } else {
      // Auto-compile sequential teaching steps representation for legacy compatibilities
      let steps = scenes.map((scene) => {
        return new TeachingStep({
          action_type: 'EXPLAIN',
          target_id: `scene_${scene.sceneNo}`,
          duration: scene.durationInFrames ? scene.durationInFrames / 30 : 6.0,
          script: scene.teacherScript || ''
        });
      });

      // Expand sequence applying reinforcement rules from selected strategy
      steps = PedagogicalStrategyEngine.processSequence(steps, pedagogy);
      
      teaching = new TeachingModel({
        lesson_goal: title,
        learning_objective: title,
        teaching_steps: steps,
        teaching_strategy: 'sequential'
      });
    }

    const Narration = require('../synchronization/models/Narration');
    const NarrationSegment = require('../synchronization/models/NarrationSegment');

    let narration = null;
    if (adaptedJson.narration) {
      narration = Narration.deserialize(adaptedJson.narration);
    } else {
      // Generate default sequential narration timeline matching scene scripts
      let currentStart = 0;
      const segments = scenes.map((scene) => {
        const duration = scene.durationInFrames ? scene.durationInFrames / 30 : 6.0;
        const segment = new NarrationSegment({
          text: scene.teacherScript || '',
          start_time: currentStart,
          estimated_duration: duration,
          related_scene: `scene_${scene.sceneNo}`
        });
        currentStart += duration;
        return segment;
      });

      narration = new Narration({
        speaker: 'Narrator',
        language: 'en',
        segments: segments
      });
    }

    // Build adapted LessonSceneGraph container
    const sceneGraph = new LessonSceneGraph(title, theme, layoutMode, scenes, metadata, teaching, narration, pedagogy);
    
    // Preserve any outer storyboard properties
    sceneGraph.lesson_id = adaptedJson.lesson_id;
    sceneGraph.lesson_uuid = adaptedJson.lesson_uuid;
    sceneGraph.book_uuid = adaptedJson.book_uuid;
    sceneGraph.subject = adaptedJson.subject;
    sceneGraph.class_name = adaptedJson.class_name;

    return sceneGraph;
  }
}

module.exports = StoryboardAdapter;
