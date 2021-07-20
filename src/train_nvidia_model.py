import models as mdls
import torch
import argparse
from training import fit_nvidia_model
import datasets as dss
import os 
import utils as ut



parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", type=str, help="Input directory for training data.")
parser.add_argument("-o", "--output", type=str, help="Output dir to save the model.")
parser.add_argument("-e", "--epochs", type=int, default=30, help="Number of epochs to train.")
parser.add_argument("-s", "--scheduler_steps", type=int, nargs="+", default=[10, 20], help="Epochs for lr decay.")
parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size.")

args = parser.parse_args()

input_dir = args.input
output = args.output
epochs = args.epochs
scheduler_steps = args.scheduler_steps
batch_size = args.batch_size


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model, opt = mdls.get_nvidia_model_sgd(dev)
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, scheduler_steps, gamma=0.1)
loss = mdls.RMSELoss()

train_dir = os.path.join(input_dir, "train_merged_frames")
valid_dir = os.path.join(input_dir, "valid_merged_frames")
test_dir = os.path.join(input_dir, "test_merged_frames")

train_ds = dss.get_frame_ds(train_dir)
valid_ds = dss.get_frame_ds(valid_dir)

shuffle = True
train_dl = dss.get_dl(train_ds, batch_size=batch_size, shuffle=shuffle)
valid_dl = dss.get_dl(valid_ds, batch_size=batch_size)


history = fit_nvidia_model(epochs, model, loss, opt, scheduler, train_dl, valid_dl, dev)

ut.save_sqrt_history_img(history, os.path.join(output, "history.png"))

mdls.save_model(model, os.path.join(output, "model.pt"))



