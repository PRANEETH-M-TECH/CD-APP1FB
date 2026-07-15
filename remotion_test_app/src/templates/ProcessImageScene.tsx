import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, AbsoluteFill, Img, staticFile } from 'remotion';
import { TransitionSeries, linearTiming } from '@remotion/transitions';
import { fade } from '@remotion/transitions/fade';
import { slide } from '@remotion/transitions/slide';
import { wipe } from '@remotion/transitions/wipe';
import { VisualStep } from '../types';
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

interface ProcessImageSceneProps {
  visual_steps: VisualStep[];
  theme: 'indigo' | 'gold' | 'emerald' | 'rose';
  background_style?: string;
  background_color?: string;
  durationInFrames?: number;
}

const THEME_ACCENTS: Record<string, string> = {
  indigo: '#6366f1',
  gold: '#fbbf24',
  emerald: '#10b981',
  rose: '#f43f5e',
};

const getPresentation = (transition: string) => {
  switch (transition) {
    case 'fade':
      return fade();
    case 'slide':
      return slide();
    case 'wipe':
      return wipe();
    default:
      return fade();
  }
};

const StepRenderer: React.FC<{ step: any; theme: string }> = ({ step, theme }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const stepDurationFrames = Math.round(step.duration_seconds * fps);

  // Compute Ken Burns scale and translation based on camera_motion
  let scale = 1.0;
  let translateX = 0;
  let translateY = 0;

  if (step.animation.camera_motion === 'zoom_in') {
    scale = interpolate(frame, [0, stepDurationFrames], [1.0, 1.25], {
      extrapolateRight: 'clamp',
    });
  } else if (step.animation.camera_motion === 'zoom_out') {
    scale = interpolate(frame, [0, stepDurationFrames], [1.25, 1.0], {
      extrapolateRight: 'clamp',
    });
  } else if (step.animation.camera_motion === 'pan_left') {
    translateX = interpolate(frame, [0, stepDurationFrames], [0, -40], {
      extrapolateRight: 'clamp',
    });
  } else if (step.animation.camera_motion === 'pan_right') {
    translateX = interpolate(frame, [0, stepDurationFrames], [0, 40], {
      extrapolateRight: 'clamp',
    });
  }

  const imageUrl = step.content.image_url || '';
  const resolvedImgSrc = imageUrl.startsWith('http') || imageUrl.startsWith('data:') 
    ? imageUrl 
    : staticFile(imageUrl.startsWith('/') ? imageUrl : `/${imageUrl}`);

  return (
    <AbsoluteFill style={{ overflow: 'hidden', justifyContent: 'center', alignItems: 'center' }}>
      {step.visual_type === 'illustration' && imageUrl ? (
        <AbsoluteFill style={{
          transform: `scale(${scale}) translate(${translateX}px, ${translateY}px)`,
        }}>
          <Img
            src={resolvedImgSrc}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover',
            }}
          />
        </AbsoluteFill>
      ) : step.visual_type === 'diagram' && step.content.svg_elements ? (
        <AbsoluteFill style={{
          justifyContent: 'center',
          alignItems: 'center',
          transform: `scale(${scale})`,
        }}>
          <svg
            viewBox="0 0 500 400"
            style={{
              width: '80%',
              height: '80%',
              maxHeight: '400px',
              maxWidth: '500px',
              backgroundColor: '#1e293b',
              borderRadius: '16px',
              border: '2px solid #334155',
              padding: '20px',
            }}
          >
            {step.content.svg_elements.map((el: any, elIdx: number) => {
              const commonStrokeProps = {
                stroke: el.stroke || '#f8fafc',
                strokeWidth: el.stroke_width || 3,
                strokeLinecap: 'round' as const,
              };

              const commonFillProps = {
                fill: el.fill || 'none',
              };

              let svgNode: React.ReactNode = null;
              switch (el.type) {
                case 'circle':
                  svgNode = <circle cx={el.cx || 0} cy={el.cy || 0} r={el.r || 20} {...commonStrokeProps} {...commonFillProps} />;
                  break;
                case 'rect':
                  svgNode = <rect x={el.x || 0} y={el.y || 0} width={el.width || 40} height={el.height || 30} rx={el.rx || 0} {...commonStrokeProps} {...commonFillProps} />;
                  break;
                case 'ellipse':
                  svgNode = <ellipse cx={el.cx || 0} cy={el.cy || 0} rx={el.rx || 30} ry={el.ry || 20} {...commonStrokeProps} {...commonFillProps} />;
                  break;
                case 'line':
                  svgNode = <line x1={el.x1 || 0} y1={el.y1 || 0} x2={el.x2 || 100} y2={el.y2 || 100} {...commonStrokeProps} />;
                  break;
                case 'path':
                  svgNode = <path d={el.d || ''} {...commonStrokeProps} {...commonFillProps} />;
                  break;
              }

              return (
                <g key={`step-el-${elIdx}`}>
                  {svgNode}
                  {el.label && (
                    <text
                      x={el.cx || el.x || (el.x1 && el.x2 ? (el.x1 + el.x2) / 2 : 0)}
                      y={(el.cy || el.y || (el.y1 && el.y2 ? (el.y1 + el.y2) / 2 : 0)) + (el.r || el.height || 30) + 18}
                      textAnchor="middle"
                      fill="#94a3b8"
                      fontSize="14"
                      fontWeight="600"
                      fontFamily="Inter, system-ui, sans-serif"
                    >
                      {el.label}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        </AbsoluteFill>
      ) : step.visual_type === 'equation' && step.content.text_content ? (
        <AbsoluteFill style={{
          justifyContent: 'center',
          alignItems: 'center',
          padding: '40px',
          transform: `scale(${scale})`,
        }}>
          <div
            style={{
              fontSize: '3.5rem',
              color: '#f8fafc',
              fontWeight: 'bold',
              fontFamily: 'Inter, system-ui, sans-serif',
              backgroundColor: 'rgba(30, 41, 59, 0.85)',
              padding: '24px 48px',
              borderRadius: '16px',
              border: '2px solid #475569',
              boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.3)',
              textAlign: 'center',
            }}
            dangerouslySetInnerHTML={{ __html: renderLaTeX(step.content.text_content) }}
          />
        </AbsoluteFill>
      ) : (
        <AbsoluteFill style={{
          justifyContent: 'center',
          alignItems: 'center',
          padding: '40px',
        }}>
          <div style={{
            fontSize: '2.5rem',
            color: '#94a3b8',
            fontStyle: 'italic',
            textAlign: 'center',
          }}>
            [Visual Step: {step.focus || step.visual_type}]
          </div>
        </AbsoluteFill>
      )}

    </AbsoluteFill>
  );
};

export const ProcessImageScene: React.FC<ProcessImageSceneProps> = ({
  visual_steps,
  theme,
  background_style,
  background_color,
  durationInFrames,
}) => {
  const { fps } = useVideoConfig();

  // Root background styles based on schema inputs
  const rootStyle: React.CSSProperties = {
    width: '100%',
    height: '100%',
    background: background_style || background_color || '#0f172a',
    overflow: 'hidden',
  };

  // Build a flat array of TransitionSeries children (no React.Fragment wrappers allowed)
  const seriesChildren: React.ReactNode[] = [];
  
  // Total frames for this scene sequence
  const totalSceneFrames = durationInFrames || (visual_steps.reduce((acc, s) => acc + s.duration_seconds, 0) * fps);
  let accumulatedFrames = 0;

  visual_steps.forEach((step, idx) => {
    const isLast = idx === visual_steps.length - 1;
    let stepDurationFrames = Math.round(step.duration_seconds * fps);

    if (isLast) {
      // Last step stretches to absorb any extra duration of the narration track
      stepDurationFrames = Math.max(30, totalSceneFrames - accumulatedFrames);
    } else {
      accumulatedFrames += stepDurationFrames;
    }

    const nextStep = visual_steps[idx + 1];

    seriesChildren.push(
      <TransitionSeries.Sequence
        key={`step-seq-${step.step_no}`}
        durationInFrames={stepDurationFrames}
      >
        <StepRenderer step={step} theme={theme} />
      </TransitionSeries.Sequence>
    );

    if (idx < visual_steps.length - 1 && nextStep) {
      seriesChildren.push(
        <TransitionSeries.Transition
          key={`step-trans-${step.step_no}`}
          presentation={getPresentation(nextStep.animation.transition) as any}
          timing={linearTiming({ durationInFrames: 15 })}
        />
      );
    }
  });

  return (
    <div style={rootStyle}>
      <TransitionSeries>
        {seriesChildren}
      </TransitionSeries>
    </div>
  );
};
