import cv2
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import os
import frames_preprocessing as fp
from custom_transform import bgr_to_rgb, Crop, BGRToYUV
import utils as ut
import hydra
from omegaconf import DictConfig, OmegaConf


# parser = argparse.ArgumentParser()
# parser.add_argument("--cpus", type=int, default=2, help="Number of cores to use.")
# parser.add_argument("-i", "--input", type=str, default=os.path.join("calib_challenge", "labeled"), help="Directory with videos.")
# parser.add_argument("-o", "--output", type=str, default="data", help="Output directory.")
# parser.add_argument("--videos", type=int, metavar="1 2 ... ", nargs='+', help="Videos to convert.")
# # parser.add_argument("-c", "--conversion_type", type=str, metavar="conversion type", help="Type of conversion: nvidia or temporal.")
# parser.add_argument("-h_div", "--height_divisor", type=float, metavar="H", default=3, help="Divisor of the height.")
# parser.add_argument("-w_div", "--width_divisor", type=float, metavar="W", default=3, help="Divisor of the width.")
# parser.add_argument("--train_split", type=float, metavar="train_split", default=0.8, help="A number between 0 and 1 for the train split.")
# parser.add_argument("--test_split", type=float, metavar="test_split", default=0.1, help="A number between 0 and 1 for the test split.")
# parser.add_argument("-m", "--merge", action="store_true", help="Merge the frames in one directory.")
# parser.add_argument("-j", "--jitter", type=float, metavar="b c s h", nargs="*", help="Specify color jitter. See pytorch doc.")
# parser.add_argument("-t", "--translate", type=float, metavar="th tw", nargs="*", help="Specify image translation. See pytorch doc.")
# parser.add_argument("-r", "--rotation", type=float, metavar="a1 a2", nargs="*", help="Specify image rotation. See pytorch doc.")
# parser.add_argument("-cr", "--crop", type=int, metavar="x1 x2 y1 y2", nargs=4, default=[25, 350, 125, 231],
#                     help="Specify image crop height.")

# args = parser.parse_args()

#TODO refactor all. The way the script is organized is a little bit a shit. Add more flexibility.


#If normalize or crop (or both) are specified, they are applied at the end of every other transformation

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
    print("Start setup videos.")
    timer.start()
    fp.setup_videos(videos, outputs, angles, selected_frames, basic_transform)
    timer.end()
    print(f"Finished setup video in {timer}")
        


if __name__ == "__main__":
    do_conversion()