import cv2
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import os
import frames_preprocessing as fp
from custom_transform import bgr_to_rgb, Crop, BGRToYUV
import utils as ut
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config")
def do_conversion(cfg: DictConfig):
    args = cfg["setup_conversion"]["conversion"]
    HEIGHT = 874 
    WIDTH = 1164
    height_div = args.height_divisor
    width_div = args.width_divisor
    new_height = int(HEIGHT // height_div)
    new_width = int(WIDTH // width_div)
        
    color_repr = BGRToYUV() if args.yuv else bgr_to_rgb   
    trf_resize = T.Resize((new_height, new_width), interpolation=InterpolationMode.BICUBIC)

    trf_crop = None
    if args.crop is not None:
        args_crop = [int(new_height * args.crop[0]), int(new_height - new_height * args.crop[1]),
                        int(new_width * args.crop[2]), int(new_width - new_width * args.crop[3])]
        trf_crop = Crop(*args_crop)

    if trf_crop is not None:
        basic_transform = T.Compose([color_repr, T.ToTensor(), trf_resize, trf_crop])
    else:
        basic_transform = T.Compose([color_repr, T.ToTensor(), trf_resize])

    
    #Basic conversion
    video_names = args.videos
    data_dir = args.output
    ut.create_dir(data_dir)
    video_dir = args.input
    selected_frames_dir = args.selected_frames

    videos = [os.path.join(video_dir, str(video_name) + ".hevc") for video_name in video_names]
    selected_frames = [os.path.join(selected_frames_dir, str(video_name) + ".txt") for video_name in video_names]
    angles = [os.path.join(video_dir, str(video_name) + ".txt") for video_name in video_names]
    outputs = [os.path.join(data_dir, str(video_name)) for video_name in video_names]
    ut.create_dirs(outputs)

    timer = ut.Timer()

    #Setup videos 
    if len(videos) > 0:
        print("Start setup videos.")
        timer.start()
        if args.single_tensor:
            fp.setup_videos_single_tensor(videos, outputs, angles, selected_frames, basic_transform)
        else:
            fp.setup_videos(videos, outputs, angles, selected_frames, basic_transform)
        timer.end()
        print(f"Finished setup video in {timer}")
    

    if args.test_video is not None:
        test_video_path = os.path.join(video_dir, str(args.test_video) + ".hevc")
        test_angles_path = os.path.join(video_dir, str(args.test_video) + ".txt")
        output = os.path.join(data_dir, str(args.test_video))
        ut.create_dir(output)
        fp.convert_test_video(test_video_path, output, test_angles_path, basic_transform)


if __name__ == "__main__":
    do_conversion()