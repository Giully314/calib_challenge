import argparse
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import os
import frames_preprocessing as fp

#TODO add time and verbose mode

parser = argparse.ArgumentParser()
parser.add_argument("--cpus", type=int, default=2, help="Number of cores to use.")
parser.add_argument("-i", "--input", type=str, default=os.path.join("calib_challenge", "labeled"), help="Directory with videos.")
parser.add_argument("-o", "--output", type=str, default="data", help="Output directory.")
parser.add_argument("--videos", type=int, metavar="1 2 ... ", nargs='+', help="Videos to convert.")
# parser.add_argument("-c", "--conversion_type", type=str, metavar="conversion type", help="Type of conversion: nvidia or temporal.")
parser.add_argument("-h_div", "--height_divisor", type=float, metavar="H", default=3, help="Divisor of the height.")
parser.add_argument("-w_div", "--width_divisor", type=float, metavar="W", default=3, help="Divisor of the width.")
parser.add_argument("--train_split", type=float, metavar="train_split", default=0.8, help="A number between 0 and 1 for the train split.")
parser.add_argument("--test_split", type=float, metavar="test_split", default=0.1, help="A number between 0 and 1 for the test split.")
parser.add_argument("-m", "--merge", type=bool, metavar="merge", default=False, help="Merge the frames in one directory.")
parser.add_argument("-j", "--jitter", type=float, metavar="b c s h", nargs="*", help="Specify color jitter. See pytorch doc.")
parser.add_argument("-t", "--translate", type=float, metavar="th tw", nargs="*", help="Specify image translation. See pytorch doc.")
parser.add_argument("-r", "--rotation", type=float, metavar="a1 a2", nargs="*", help="Specify image rotation. See pytorch doc.")
parser.add_argument("-cr", "--crop", type=int, metavar="x1 x2 y1 y2", nargs=4, default=[25, 350, 125, 231],
                    help="Specify image crop height.")

args = parser.parse_args()

# print(args)

conversion_type = args.conversion_type

num_of_cpu = args.cpus
merge = args.merge
HEIGHT = 874 
WIDTH = 1164
height_div = args.height_divisor
width_div = args.width_divisor
new_height = HEIGHT // height_div
new_width = WIDTH // width_div

train_split = args.train_split
test_split = args.test_split


trf_resize = T.Resize((new_height, new_width), interpolation=InterpolationMode.BICUBIC)
basic_transform = T.Compose([fp.bgr_to_rgb, T.ToTensor(), trf_resize])


trf_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)


trf_rotation = None
trf_jitter = None
trf_translate = None

if args.rotation is not None:
    a1 = args.rotation[0]
    a2 = args.rotation[1]
    rotate = T.RandomRotation((a1, a2), interpolation=InterpolationMode.BILINEAR)
    trf_rotation = []
    trf_rotation.append(rotate)


if args.jitter is not None:
    brightness = args.jitter[0]
    contrast = args.jitter[1]
    saturation = args.jitter[2]
    hue = args.jitter[3]
    jitter = T.ColorJitter(brightness, contrast, saturation, hue)
    trf_jitter = []
    trf_jitter.append(jitter)


if args.translate is not None:
    translate_x = args.translate[0]
    translate_y = args.translate[1]
    translate = T.RandomAffine(0, (translate_x, translate_y), interpolation=InterpolationMode.BILINEAR)
    trf_translate = []
    trf_translate.append(translate)


crop_x1 = args.crop[0]
crop_x2 = args.crop[1]
crop_y1 = args.crop[2]
crop_y2 = args.crop[3]
trf_crop = fp.Crop(crop_x1, crop_x2, crop_y1, crop_y2)


trf_standard = T.Compose([trf_crop, trf_normalize])


if trf_rotation is not None:
    trf_rotation.append(trf_crop)
    trf_rotation.append(trf_normalize)
    trf_rotation = T.Compose(trf_rotation)

if trf_jitter is not None:
    trf_jitter.append(trf_crop)
    trf_jitter.append(trf_normalize)
    trf_jitter = T.Compose(trf_jitter)

if trf_translate is not None:
    trf_translate.append(trf_crop)
    trf_translate.append(trf_normalize)
    trf_translate = T.Compose(trf_translate)


#Basic conversion
video_names = args.videos
data_dir = args.output
fp.create_dir(data_dir)
video_dir = args.input

videos = [os.path.join(video_dir, str(video_name) + ".hevc") for video_name in video_names]
angles = [os.path.join(video_dir, str(video_name) + ".txt") for video_name in video_names]
outputs = [os.path.join(data_dir, str(video_name)) for video_name in video_names]
fp.create_dirs(outputs)

fp.setup_videos(videos, outputs, angles, basic_transform, num_of_cpu=num_of_cpu)


#Split dataset
fp.split_train_valid_test(outputs, data_dir, train_split, test_split, num_of_cpu=num_of_cpu)


#augment data
basic_train_dir = os.path.join(data_dir, "basic_train")
inputs = [os.path.join(basic_train_dir, video_name) for video_name in video_names]
train_dir = os.path.join(data_dir, "train")
fp.create_dir(train_dir)

count = 0
standard_dirs = [os.path.join(train_dir, str(video_name)) for video_name in video_names]
fp.create_dirs(standard_dirs)
fp.augment_videos(inputs, standard_dirs, trf_standard, num_of_cpu=num_of_cpu)
count += len(standard_dirs)

merge_dirs = [*standard_dirs] #for merge

if trf_jitter is not None:
    jitter_dirs = [os.path.join(train_dir, str(video_name + count)) for video_name in video_names]
    merge_dirs += jitter_dirs
    fp.create_dirs(jitter_dirs)
    fp.augment_videos(inputs, jitter_dirs, trf_jitter, num_of_cpu=num_of_cpu)
    count += len(jitter_dirs)

if trf_rotation is not None:
    rotation_dirs = [os.path.join(train_dir, str(video_name + count)) for video_name in video_names]
    merge_dirs += rotation_dirs
    fp.create_dirs(rotation_dirs)
    fp.augment_videos(inputs, rotation_dirs, trf_rotation, num_of_cpu=num_of_cpu)
    count += len(rotation_dirs)

if trf_translate is not None:
    translate_dirs = [os.path.join(train_dir, str(video_name + count)) for video_name in video_names]
    merge_dirs += translate_dirs
    fp.create_dirs(translate_dirs)
    fp.augment_videos(inputs, translate_dirs, trf_translate, num_of_cpu=num_of_cpu)
    count += len(translate_dirs)


#normalize validation and test set
valid_dir = os.path.join(data_dir, "valid")
valid_dirs = [os.path.join(valid_dir, d) for d in os.listdir(valid_dir) if os.path.isdir(d)]
fp.augment_videos(valid_dirs, valid_dirs, trf_standard, num_of_cpu=num_of_cpu)

test_dir = os.path.join(data_dir, "test")
test_dirs = [os.path.join(test_dir, d) for d in os.listdir(test_dir) if os.path.isdir(d)]
fp.augment_videos(test_dirs, test_dirs, trf_standard, num_of_cpu=num_of_cpu)


if merge:
    #TODO remove empty directories.
    fp.merge_frames(merge_dirs, os.path.join(data_dir, "train_merged_frames"))
    fp.merge_frames(valid_dirs, os.path.join(data_dir), "valid_merged_frames")
    fp.merge_frames(test_dirs, os.path.join(data_dir), "test_merged_frames")


