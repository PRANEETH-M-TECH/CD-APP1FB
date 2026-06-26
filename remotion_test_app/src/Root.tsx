import React from 'react';
import { Composition } from 'remotion';
import { ConveyorComposition } from './Composition';
import { getAudioDurationInSeconds } from '@remotion/media-utils';
import { staticFile, getInputProps } from 'remotion';
import sampleStoryboard from './sample_storyboard.json';
import { Storyboard, Scene } from './types';

export const RemotionRoot: React.FC = () => {
  const inputProps = getInputProps() as any;
  const storyboard = Object.keys(inputProps).length > 0 ? inputProps : sampleStoryboard;

  return (
    <>
      <Composition
        id="StoryboardVideo"
        component={ConveyorComposition as any}
        fps={30}
        width={1280}
        height={720}
        defaultProps={storyboard as Storyboard}
        calculateMetadata={async ({ props }: any) => {
          const scenes = props.scenes || [];
          const scenesWithDuration = await Promise.all(
            scenes.map(async (scene: Scene) => {
              let durationInSeconds = 6; // default fallback
              if (scene.audio_url) {
                try {
                  const audioPath = staticFile(scene.audio_url);
                  durationInSeconds = await getAudioDurationInSeconds(audioPath);
                } catch (e) {
                  console.warn("Failed to get audio duration for", scene.audio_url, "falling back to estimation:", e);
                  const words = scene.teacher_script ? scene.teacher_script.split(/\s+/).length : 0;
                  durationInSeconds = Math.max(5, words * 0.45 + 1.5);
                }
              } else {
                const words = scene.teacher_script ? scene.teacher_script.split(/\s+/).length : 0;
                durationInSeconds = Math.max(5, words * 0.45 + 1.5);
              }

              const fps = 30;
              const durationInFrames = Math.ceil(durationInSeconds * fps);
              return {
                ...scene,
                durationInFrames,
              };
            })
          );

          const totalDurationInFrames = scenesWithDuration.reduce(
            (acc: number, scene: Scene) => acc + (scene.durationInFrames || 180),
            0
          );

          return {
            durationInFrames: Math.max(30, totalDurationInFrames),
            props: {
              ...props,
              scenes: scenesWithDuration,
            } as any,
          };
        }}
      />
    </>
  );
};
