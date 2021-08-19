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

@hydra.main(config_path="config", config_name="windows_nvidia_setup.yaml")
def do_conversion(cfg: DictConfig):
    args = cfg["conversion"]
    num_of_cpu = args.cpus
    merge = args.merge
    HEIGHT = 874 
    WIDTH = 1164
    height_div = args.height_divisor
    width_div = args.width_divisor
    new_height = int(HEIGHT // height_div)
    new_width = int(WIDTH // width_div)

    train_split = args.train_split
    test_split = args.test_split

        
    color_repr = BGRToYUV() if args.yuv else bgr_to_rgb   
    trf_resize = T.Resize((new_height, new_width), interpolation=InterpolationMode.BICUBIC)
    basic_transform = T.Compose([color_repr, T.ToTensor(), trf_resize])

    #Basic conversion
    video_names = args.videos
    data_dir = args.output
    ut.create_dir(data_dir)
    video_dir = args.input

    videos = [os.path.join(video_dir, str(video_name) + ".hevc") for video_name in video_names]
    angles = [os.path.join(video_dir, str(video_name) + ".txt") for video_name in video_names]
    outputs = [os.path.join(data_dir, str(video_name)) for video_name in video_names]
    ut.create_dirs(outputs)

    timer = ut.Timer()

    #Setup videos 
    print("Start setup videos.")
    timer.start()
    fp.setup_videos(videos, outputs, angles, basic_transform, num_of_cpu=num_of_cpu)
    timer.end()
    print(f"Finished setup video in {timer}")


    #Split dataset
    print("Start split data.")
    timer.start()
    fp.split_train_valid_test(outputs, data_dir, train_split, test_split, num_of_cpu=num_of_cpu)
    timer.end()
    print(f"Finished split data in {timer}.")

    transformations = {}

    if args.normalization is not None:
        trf_normalize = T.Normalize(mean=args.normalization[0], std=args.normalizations[1], inplace=True)
        transformations["normalization"] = trf_normalize

    if args.rotation is not None:
        a1 = args.rotation[0]
        a2 = args.rotation[1]
        trf_rotation = T.RandomRotation((a1, a2), interpolation=InterpolationMode.BILINEAR)
        transformations["rotation"] = trf_rotation

    if args.jitter is not None:
        brightness = tuple(args.jitter[0])
        contrast = tuple(args.jitter[1])
        saturation = tuple(args.jitter[2])
        hue = tuple(args.jitter[3])
        trf_jitter = T.ColorJitter(brightness, contrast, saturation, hue)
        transformations["jitter"] = trf_jitter

    if args.translate is not None:
        translate_x = args.translate[0]
        translate_y = args.translate[1]
        trf_translate = T.RandomAffine(0, (translate_x, translate_y), interpolation=InterpolationMode.BILINEAR)
        transformations["translate"] = trf_translate

    if args.crop is not None:
        crop_y1 = new_height * args.crop[0]
        crop_y2 = new_height - new_height * args.crop[1]
        crop_x1 = new_width * args.crop[2]
        crop_x2 = new_width - new_width * args.crop[3]
        trf_crop = Crop(crop_x1, crop_x2, crop_y1, crop_y2)
        transformations["crop"] = trf_crop

    #augment data
    basic_train_dir = os.path.join(data_dir, "basic_train")
    train_dir = os.path.join(data_dir, "train")
    ut.create_dir(train_dir)
    inputs = [os.path.join(basic_train_dir, str(video_name)) for video_name in video_names]
    for transform, output in zip(args.transformations, args.aug_data_output):
        print(f"Start augment frames with {transform}.")
        timer.start()
        output = os.path.join(train_dir, output)
        ut.create_dir(output)
        output_aug_data = [os.path.join(output, str(video_name)) for video_name in video_names]
        ut.create_dirs(output_aug_data)
        trfs = [transformations[t] for t in transform]
        full_transform = T.Compose(trfs)
        fp.augment_videos(inputs, output_aug_data, full_transform, num_of_cpu)
        timer.end()
        print(f"Finished augment frames with {transform} in {timer}.")

    #validation/test set preprocessing only with crop and normalize (if specified)
    #TODO for now i specify only the crop for preprocessing. Write a better generalized version

    if "crop" in args.transformations:
        print("Start cropping validation set.")
        timer.start()
        valid_dir = os.path.join(data_dir, "valid")
        valid_dirs = [os.path.join(valid_dir, str(video_name)) for video_name in video_names]
        fp.augment_videos(valid_dirs, valid_dirs, transform["crop"], num_of_cpu=num_of_cpu)
        timer.end()
        print(f"Finished cropping validation set in {timer}")

        print("Start cropping test set.")
        timer.start()
        test_dir = os.path.join(data_dir, "test")
        test_dirs = [os.path.join(test_dir, str(video_name)) for video_name in video_names]
        fp.augment_videos(test_dirs, test_dirs, transform["crop"], num_of_cpu=num_of_cpu)
        timer.end()
        print(f"Finished cropping test set in {timer}")

    #TODO check if valid/test set needs some sort of aug preprocessing.
    #normalize validation and test set
    # print("Start normalize validation set.")
    # timer.start()
    # valid_dir = os.path.join(data_dir, "valid")
    # valid_dirs = [os.path.join(valid_dir, str(video_name)) for video_name in video_names]
    # print(valid_dirs)
    # fp.augment_videos(valid_dirs, valid_dirs, trf_standard, num_of_cpu=num_of_cpu)
    # timer.end()
    # print(f"Finished normalize validation set in {timer}")

    # print("Start normalize test set.")
    # timer.start()
    # test_dir = os.path.join(data_dir, "test")
    # test_dirs = [os.path.join(test_dir, str(video_name)) for video_name in video_names]
    # fp.augment_videos(test_dirs, test_dirs, trf_standard, num_of_cpu=num_of_cpu)
    # timer.end()
    # print(f"Finished normalize test set in {timer}")

    ut.delete_dirs(outputs)

    #TODO refactor merge.
    # if merge:
    #     print("Start merge frames.")
    #     timer.start()
    #     #TODO remove empty directories.
    #     fp.merge_frames(merge_dirs, os.path.join(data_dir, "train_merged_frames"))
    #     fp.merge_frames(valid_dirs, os.path.join(data_dir), "valid_merged_frames")
    #     fp.merge_frames(test_dirs, os.path.join(data_dir), "test_merged_frames")
    #     timer.end()
    #     print(f"Finished merge frames in {timer}.")

    #     ut.delete_dir(test_dir)
    #     ut.delete_dir(valid_dir)
    #     ut.delete_dir(basic_train_dir)
    #     ut.delete_dirs(standard_dirs)

    #     if trf_translate is not None:
    #         ut.delete_dirs(translate_dirs)
        
    #     if trf_jitter is not None:
    #         ut.delete_dirs(jitter_dirs)

    #     if trf_rotation is not None:
    #         ut.delete_dirs(rotation_dirs)
        



if __name__ == "__main__":
    do_conversion()