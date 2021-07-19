import argparse
import models as mdls
import os
import datasets as dss
import utils as ut

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Input directory.")
parser.add_argument("-o", "--output", type=str, help="Output directory for angles results.")
parser.add_argument("-m", "--model_params_path", type=str, help="Path of the model's parameters for inference.")

args = parser.parse_args()

input_dir = args.input
output_dir = args.output
model_dict_path = args.model_params_path

video_dirs = [os.path.join(input_dir, video) for video in os.listdir(input_dir) if os.path.isdir(video)]
output_angles = [os.path.join(output_dir, video + ".txt") for video in os.listdir(input_dir) if os.path.isdir(video)]


model = mdls.load_nvidia_model(path=model_dict_path)    

for video_dir, output_file in zip(video_dirs, output_angles):
    ds = dss.get_frame_ds(video_dir)
    dl = dss.get_dl(ds, batch_size=128)
    ut.inference_and_save(model, dl, output_file)







