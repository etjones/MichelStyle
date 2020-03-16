#! /usr/bin/env python

'''
MichelStyle, made by Remideza for Bend the Future single "Otaniemi"
'''
import argparse
import os
from pathlib import Path
import sys
import time

# TODO: these imports take a long time. Import inline to decrease startup time? 
import cv2
import tensorflow as tf
import numpy as np
import tensorflow_hub
import moviepy
import PIL

# ==============
# = TYPE HINTS =
# ==============
from typing import List, Dict, Sequence, Tuple, Any
class CVImage(np.ndarray): pass 

def main():
    args = parse_all_args()
    total_elapsed = style_video(source_dir_or_image=args.source, styles_dir=args.styles, output_dir=args.output_frames)    
    print(f'Wrote frames to {args.output_frames}')

    if args.video:
        vid_start = time.time()
        output_path = write_video_file(args.frames, output_path=args.video, 
                        fps=args.fps, audio_path=args.audio)
        total_elapsed += (time.time() - vid_start)
    print(f'Total time: {total_elapsed/60:.1f} minutes')

def parse_all_args(args_in=None):
    ''' Set up argparser and return a namespace with named
    values from the command line arguments.  
    If help is requested (-h / --help) the help message will be printed 
    and the program will exit.
    '''
    program_description = '''Output a video with styles transferred onto each frame'''

    parser = argparse.ArgumentParser( description=program_description,
                formatter_class=argparse.HelpFormatter)

    # Replace these with your arguments below
    parser.add_argument('--source', type=Path, required=True,
        help=('A directory containing frames of a movie sequence that should have '
        'styles applied to them. Frame numbers start at 1. A single image may also '
        'be supplied'))

    parser.add_argument( '--styles', type=Path, required=True,
        help=('A directory containing image files to take styles from. Each '
        'image should have a number for the frame it should be most applied to '))

    parser.add_argument('-o','--output_frames', type=Path, default=Path('styled_frames'),
        help='Path to an output directory where stylized frames will be written.  Default: "%(default)s"')
    
    parser.add_argument('-v', '--video', type=Path, 
        help='Path to an MP4 output file.')

    parser.add_argument('-a', '--audio', type=Path, default=None,
        help='Path to an  MP3 file. If specified, it will be added to the '
        'generated video')

    parser.add_argument('-f', '--fps', type=int, default=24,
        help='Destination frame rate. Default: %(default)s')

    # TODO: support this option
    # parser.add_argument('--force_overwrite', action='store_true', default=False,
    #     help=('If specified, force already-written files in the OUTPUT_FRAMES directory '
    #     'to be overwritten' ))

    # print help if no args are supplied
    if len(sys.argv) <= 2: 
        sys.argv.append('--help')

    # If args_in isn't specified, args will be taken from sys.argv
    args = parser.parse_args(args_in)

    # Validation:
    if not args.source.exists():
        raise 

    ensure_dir(args.output_frames)

    return args

def style_video(source_dir_or_image: Path, 
                styles_dir: Path, 
                output_dir:Path) -> float:
    total_start_time = time.time()
    params = calculate_styling_params(source_dir_or_image, styles_dir)
    print('Transferring styles...\n\n')
    hub_module = get_tf_hub()

    frame_count = len(params)
    style_images: Dict[Path, CVImage] = {}

    single_source_file = is_image(source_dir_or_image)
    if single_source_file:
        source_image = frame_image(source_dir_or_image, as_float=True)

    for frame, (source_path, style_a_path, style_b_path, style_ratio) in params.items():
        start_time = time.time()
        output_path = output_dir / f'{frame}.jpg'

        if not single_source_file:
            source_image = frame_image(source_path, as_float=True)

        style_a_image = style_images.setdefault(style_a_path, frame_image(style_a_path))
        style_b_image = style_images.setdefault(style_b_path, frame_image(style_a_path))

        stylized_image = transfer_styles(source_image, style_a_image, style_b_image, style_ratio, hub_module)
        stylized_image.save(output_path)

        infoscreen(frame, frame_count, time.time() - start_time)

    return time.time() - total_start_time

def calculate_styling_params(source_dir_or_image: Path, 
                             styles_dir: Path,) -> Dict[int, Tuple[Path, Path, Path, float]]:
    params: Dict[int, Tuple[Path, Path, Path, float]] = {}

    # Figure out how many frames we'll need
    source_frame_paths = numbered_images_dict(source_dir_or_image)
    style_frame_paths = numbered_images_dict(styles_dir)

    style_frame_numbers = sorted(style_frame_paths.keys())
    source_frame_numbers = sorted(source_frame_paths.keys())

    first_source_frame, last_source_frame = source_frame_numbers[0], source_frame_numbers[-1]
    first_style_frame, last_style_frame = style_frame_numbers[0], style_frame_numbers[-1]

    style_args: Dict[int, Tuple[Path, Path, float]]= {}

    # TODO: get frame lengths from movies, too. 
    # TODO: Handle missing source frames, e.g. 1.jpg & 3.jpg exist, but not 2.jpg
    frame_count = last_style_frame

    if len(source_frame_numbers) == 1:
        source_path = source_frame_paths[first_source_frame]
        source_frame_paths = dict({f: source_path for f in range(1,last_style_frame + 1)})

    # Insert beginning and end elements in the style transitions so the 
    # entire frame range is covered by a pair of style images
    if first_style_frame != 1:
        style_frame_paths[1] = style_frame_paths[first_style_frame]

    if last_style_frame != frame_count:
        style_frame_paths[frame_count] = style_frame_paths[last_style_frame]

    style_transitions = sorted(list(style_frame_paths.keys()))
    transition_pairs = zip(style_transitions[:-1], style_transitions[1:])

    for start_frame, end_frame in transition_pairs:
        style_a_path = style_frame_paths[start_frame]
        style_b_path = style_frame_paths[end_frame]

        for frame in range(start_frame, end_frame + 1):
            # if frame == start_frame, we will have just calculated its params
            # for the previous start_frame/end_frame pair; skip this step
            if frame in params:
                continue
            style_ratio = (frame - start_frame)/(end_frame - start_frame)
            params[frame] = (source_frame_paths[frame], style_a_path, style_b_path, style_ratio)

    return params

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
def get_tf_hub():
    TF_HUB = tensorflow_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    # importing hub_module prints some debug info to my screen. Remove that
    clear_screen()
    return TF_HUB

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def ensure_dir(dir_name:Path) -> bool:
    if dir_name.exists():
        if dir_name.is_dir():
            return True
        else:
            raise ValueError(f'{dir_name} exists, but is not a directory')
    else:
        dir_name.mkdir()
        return True

def clear_screen():
    if sys.platform == 'darwin':
        os.system('clear')
    else: 
        os.system('cls')

def infoscreen(frame: int, total_frames:int, frame_elapsed: float):
    minutes_left = (frame_elapsed * (total_frames - frame))/60

    line = f"   SimpleStyleTransfer - {frame}/{total_frames}"
    marquee = '='*(len(line) + 3)

    clear_screen()
    print(marquee)
    print(line)
    print(marquee)
    print()
    # print(f"------------------------------------")
    # print(f"   SimpleStyleTransfer - {frame}/{total_frames}")
    # print(f"------------------------------------")
    print()
    print(f"{minutes_left:.1f} minutes remaining")

def is_image(image:Path) -> bool:
    IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.gif')

    return image.exists() and image.is_file() and image.suffix in IMAGE_SUFFIXES

def numbered_images_dict(a_dir: Path) -> Dict[int, Path]:
    result: Dict[int, Path] = {}

    # If a_dir is an image file, not a directory, we'll just return a single 
    # element dict
    if is_image(a_dir):
        result = {1: a_dir}
    elif a_dir.is_dir():
        for f in a_dir.iterdir():
            # TODO: maybe accept files with a number anyplace in the stem?
            if f.stem.isdigit() and is_image(f):
                result[int(f.stem)] = f
    else:
        raise ValueError(f'argument {a_dir} is neither a directory nor an image '
                        'file we know how to handle')
                
    return result

def frame_image(path:Path, as_float:bool = False) -> CVImage: 
    img = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    if as_float:
        img = img.astype(np.float32)[np.newaxis, ...] / 255.0
    return img


if __name__ == '__main__':
    main()