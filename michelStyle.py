#! /usr/bin/env python

'''
MichelStyle, made by Remideza for Bend the Future single "Otaniemi"
'''
import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import tensorflow as tf
import numpy as np
import tensorflow_hub
import moviepy
import PIL

from typing import List, Dict, Sequence, Tuple
CVImage = np.ndarray # Mypy throws:  error: Variable "michelStyle.CVImageX" is not valid as a type
                     # TODO: figure out how to get around it
# ===========
# = GLOBALS =
# ===========
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.gif')
MAXBATCH = 100



TF_HUB = None
def get_tf_hub():
    global TF_HUB
    if not TF_HUB:
        TF_HUB = tensorflow_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    return TF_HUB

def main():
    args = parse_all_args()
    style_video(source_dir_or_image=args.source, styles_dir=args.styles, output_dir=args.frames)
    print(f'Wrote frames to {args.frames}')

    if not args.no_video:
        output_path = write_video_file(args.frames, output_path=args.output, 
                        fps=args.fps, audio_path=args.audio)

def parse_all_args(args_in=None):
    ''' Set up argparser and return a namespace with named
    values from the command line arguments.  
    If help is requested (-h / --help) the help message will be printed 
    and the program will exit.
    '''
    program_description = '''Output a video with styles transferred onto each image'''

    parser = argparse.ArgumentParser( description=program_description,
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Replace these with your arguments below
    parser.add_argument( '--styles', type=Path, required=True,
        help=('A directory containing image files to take styles from. Each '
        'image should have a number for the frame it should be most applied to '))
    parser.add_argument('--source', type=Path, required=True,
        help=('A directory containing frames of a movie sequence that should have '
        'styles applied to them. Frame numbers start at 1. A single image may also '
        'be supplied'))
    
    parser.add_argument('-o', '--output', type=Path, default=Path('video_out.mp4'),
        help='Path to output .mp4 to. Default: %(default)s')
    parser.add_argument('--frames', type=Path, default=Path('stylized_frames'),
        help='Path to a directory where stylized frames will be written. ')
    parser.add_argument('--no_video', action='store_true', default=False,
        help='Don\'t write a video file, just stylized frames to --OUTPUT_FRAMES')
    parser.add_argument('-f', '--fps', type=int, default=24,
        help='Destination frame rate. Default: %(default)s')
    parser.add_argument('-a', '--audio', type=Path, default=None,
        help='Path to an  MP3 file. If specified, it will be added to the '
        'generated video')
    parser.add_argument('--force_overwrite', action='store_true', default=False,
        help=('If specified, force already-written files in the OUTPUT_FRAMES directory '
        'to be overwritten' ))

    # If args_in isn't specified, args will be taken from sys.argv
    args = parser.parse_args(args_in)

    # Validation:
    # if not args.source.exists()

    return args

def style_video(source_dir_or_image: Path, 
                styles_dir: Path, 
                output_dir:Path=None) -> bool:
    
    hub_module = get_tf_hub()

    # Figure out how many frames we'll need
    if source_dir_or_image.is_file(): # assume source_dir_or_image is a single image file
        source_frame_paths = {1: source_dir_or_image}
    else:
        source_frame_paths = numbered_images_dict(source_dir_or_image)
    style_frame_paths = numbered_images_dict(styles_dir)

    style_frame_numbers = sorted(style_frame_paths.keys())
    source_frame_numbers = sorted(source_frame_paths.keys())

    first_source_frame, last_source_frame = source_frame_numbers[0], source_frame_numbers[-1]
    first_style_frame, last_style_frame = style_frame_numbers[0], style_frame_numbers[-1]

    style_args: Dict[int, Tuple[Path, Path, float]]= {}

    # If we have fewer frames than we do styles what do we do? 
    # This is what Remi originally did, with a single frame looping through
    # several styles
    # TODO: get frame lengths from movies, too. 
    # TODO: Handle missing source frames, e.g. 1.jpg & 3.jpg exist, but not 2.jpg
    frame_count = max(last_source_frame, last_style_frame)

    # Insert beginning and end elements in the style transitions so the 
    # entire frame range is covered by a pair of style images
    if first_style_frame != 1:
        style_frame_paths[1] = style_frame_paths[first_style_frame]

    if last_style_frame != frame_count:
        style_frame_paths[frame_count] = style_frame_paths[last_style_frame]

    style_transitions = sorted(list(style_frame_paths.keys()))
    transition_pairs = list(zip(style_transitions[:-1], style_transitions[1:]))

    # Make sure we have styles for the initial and final frames
    # TODO: we want something like writing an 0.jpg or <large_n>.jpg
    # with the contents of the first/last style frames
    which_transition = 0
    start_frame, end_frame = transition_pairs[which_transition]
    style_start_image = frame_image(style_frame_paths[start_frame])
    style_end_image = frame_image(style_frame_paths[end_frame])

    style_end_image: CVImage
    style_ratio: float
    stylized_image: CVImage  

    single_source_frame = (len(source_frame_numbers) == 1)
    if single_source_frame:
        source_image = frame_image(source_frame_paths[first_source_frame], as_float=True)

    for i in range(1, frame_count + 1):

        start_time = time.time()
        start_frame, end_frame = transition_pairs[which_transition]

        # Only load style images from disk when we change the pairs of style images
        # if i == end_frame and which_transition < len(transition_pairs) - 1:
        if i > end_frame:
            which_transition += 1
            start_frame, end_frame = transition_pairs[which_transition]
            style_start_image = frame_image(style_frame_paths[start_frame])
            style_end_image   = frame_image(style_frame_paths[end_frame])
        
        style_ratio = (i - start_frame)/(end_frame - start_frame)

        # FIXME: We currently assume we have either: 
        # a) more source frames than styles, or 
        # b) a single source frame. 
        # TODO: object if style frames have a larger value than the number
        # of source frames
        # If there are multiple source frames, but more style frames, we'd 
        # be scaling the original video in some problematic ways, so skip that for now
        if not single_source_frame:
            source_image = frame_image(source_frame_paths[i], as_float=True)

        # TODO: Currently, we don't check whether this image might already have
        # been written. That could speed up development, if for example, you
        # were repeatedly running this program to sync up style changes with
        # music, say. -ETJ 13 March 2020
        stylized_image = transfer_styles(source_image, style_start_image, style_end_image, style_ratio, hub_module)
        output_path = output_dir / f'{i}.jpg'
        stylized_image.save(output_path)

        frames_to_go = frame_count - i
        estimated_minutes_remaining = ((time.time() - start_time) * frames_to_go)/60
        infoscreen(frames_to_go, estimated_minutes_remaining)

    return True

def transfer_styles(source_image_as_floats:CVImage, 
                    style_a:CVImage, 
                    style_b: CVImage=None, 
                    style_ratio:float=0,
                    hub_module: Any=None) -> CVImage:
    # Style source_image_as_floats with a single other image, or with an affine
    # combination of two images. 
    # Note that style_ratio should be in [0,1] and represents the 
    # proportion of ** style_b ** used in the styling. 
    hub_module = hub_module or get_tf_hub()

    stylized_image: CVImage 

    if style_b is not None and style_ratio != 0:
        style_image = cv2.addWeighted(style_a, 1-style_ratio, style_b, style_ratio, 0.0)
    else:
        style_image = style_a
    style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB) 
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.0

    outputs = hub_module(tf.constant(source_image_as_floats), tf.constant(style_image))
    stylized_image = tensor_to_image(outputs[0])
    return stylized_image

def write_video_file(frames_dir: Path, output_path: Path=None, fps=24, audio_path:Path=None) -> Path:
    '''
    Writes all the numbered frames in frames_dir to an mp4
    '''
    if output_path is None:
        output_path = Path(__file__) / 'video_out.mp4'
    output_path = output_path.resolve().with_suffix('.mp4')
    out_posix = output_path.as_posix()

    frames_paths = numbered_images_dict(frames_dir)
    # assume sizes of all generated frames are the same, and get size from 
    # a random frame
    random_frame = list(frames_paths.values())[0].as_posix()

    HEIGHT, WIDTH, channels = cv2.imread(random_frame, 1).shape
    OUTPUT_SIZE = (WIDTH * 2,HEIGHT * 2)
    video = cv2.VideoWriter(out_posix, cv2.VideoWriter_fourcc(*"mp4v"), fps, OUTPUT_SIZE)
    print("Compiling video ...")
    frames_count = len(frames_paths)
    for frame_num in sorted(frames_paths.keys()):
        frame_path = frames_paths[frame_num].as_posix()
        sys.stdout.flush()
        sys.stdout.write(f'{frame_num}/{frames_count}\r')
        video.write(cv2.resize(cv2.imread(frame_path, 1), OUTPUT_SIZE))
    video.release()

    if audio_path is not None:
        mp_video = moviepy.editor.VideoFileClip(out_posix, fps)
        mp_video.write_videofile(out_posix, audio=audio_path.as_posix())

    print(f'Wrote {frames_count} frames to video at {output_path}')

    return output_path

# ===========
# = HELPERS =
# ===========

def ensure_dir(dir_name:Path) -> bool:
    if dir_name.exists():
        if dir_name.is_dir():
            return True
        else:
            raise ValueError(f'{dir_name} exists, but is not a directory')
    else:
        dir_name.mkdir()
        return True

def clear():
    if sys.platform == 'darwin':
        os.system('clear')
    else: 
        os.system('cls')

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def infoscreen(remaining_frames: int, minutes_left: float):
    clear()
    print("--------------------------")
    print("  MichelStyle - Remideza  ")
    print("--------------------------")
    print()
    print(f"{remaining_frames} remaining frames to work on")
    print(f"{minutes_left:.1f} minutes remaining")

def is_image(image:Path) -> bool:
    return image.exists() and image.is_file() and image.suffix in IMAGE_SUFFIXES

def numbered_images_dict(a_dir: Path) -> Dict[int, Path]:
    result: Dict[int, Path] = {}
    for f in a_dir.iterdir():
        # TODO: maybe accept files with a number anyplace in the stem?
        if f.stem.isdigit() and is_image(f):
            result[int(f.stem)] = f
    return result

def frame_image(path:Path, as_float:bool = False) -> CVImage: 
    img = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    if as_float:
        img = img.astype(np.float32)[np.newaxis, ...] / 255.0
    return img

if __name__ == '__main__':
    main()