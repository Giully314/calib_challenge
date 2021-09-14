from torchvision.transforms.functional import InterpolationMode
from models import CalibModel, CalibParams, RMSELoss
from torch.utils.data import DataLoader
import torch
from datasets import ConsecutiveFramesDataset, VideoDataset, DiskVideoDataset, get_consecutive_frames_ds, get_disk_consecutive_frames_ds
import training as trn
from custom_transform import CropVideo, ToOpenCV, RGBtoBGR
import torchvision.transforms as T
import os 
import numpy as np
from torch import nn
import random 
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import Timer
from informations import History, GradientFlowVisualization, ActivationMapVisualization, TestModel
import cv2

#TODO ADD LOGGING
#TODO WRITE BETTER IMPLEMENTATION FOR CONFIG TRANSFORMATIONS ON FLY.

@hydra.main(config_path="config")
def main(cfg: DictConfig):
    args = cfg["training"]["train"]
    
    img_size = tuple(args.frame_size)

    timer = Timer()

    if args.deterministic:
        seed = 1729
        def reset_rand_seed(): 
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        reset_rand_seed()
        #torch.use_deterministic_algorithms(True)

    if not args.debug:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(enabled=False)


    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"DEVICE {dev}")
    #torch.cuda.empty_cache() ??

    #NOTE: if the VideoDataset is used, make sure to shuffle these lists, for a better random behavior.
    #TODO: maybe divide the videos into equal small size (like 256 frames per block).
    videos_path = [os.path.join(training_dir, os.path.join(str(video), str(part))) 
                    for training_dir in args.training_dirs
                    for video, video_parts in zip(args.train_videos, args.videos_parts) for part in video_parts]


    if args.shuffle_parts:
        random.shuffle(videos_path)

    if args.debug:
        for video in videos_path:
            print(video)


    train_videos = [video + ".pt" for video in videos_path]
    train_angles = [video + ".txt" for video in videos_path]


    trf = cfg["training"]["transformations"]
    transformations = None
    if trf.transforms is not None:
        if trf.color_jitter:
            trf_jitter = T.ColorJitter(tuple(trf.brightness), tuple(trf.contrast), trf.saturation, trf.hue)
        if trf.rotation:
            trf_rotation = T.RandomRotation(tuple(trf.degrees), interpolation=InterpolationMode.BILINEAR)
        if trf.translate:
            trf_translate = T.RandomAffine(0, translate=tuple(trf.translations), interpolation=InterpolationMode.BILINEAR)
        if trf.crop:
            trf_crop = CropVideo(*list(trf.crop_args))

        transformations = []
        for transforms in trf.transforms:
            t = []
            for transform in transforms:
                if transform == "rotation":
                    t.append(trf_rotation)
                if transform == "crop":
                    t.append(trf_crop)
                if transform == "color_jitter":
                    t.append(trf_jitter)
                if transform == "translate":
                    t.append(trf_translate)
            
            transformations.append(T.Compose(t))



    print("Loading datasets...", end=" ")
    timer.start()
    if args.dataset == "diskvideods":
        train_dss = [DiskVideoDataset(videos_path, args.consecutive_frames, args.skips, transformations)]
    elif args.dataset == "videods":
        train_dss = [VideoDataset(train_videos, train_angles, args.consecutive_frames, args.skips, transformations)]
    timer.end()
    print(f"{timer}")


    if args.range is not None:
        start = args.range[0]
        end = args.range[1]
        for ds in train_dss:
            ds.frames = ds.frames[start:end]
            ds.angles = ds.angles[start:end]


    train_dls = [
        DataLoader(train_ds, args.batch_size,
        num_workers=args.train_workers, shuffle=args.shuffle, pin_memory=True, persistent_workers=args.persistent_workers) 
        for train_ds in train_dss]

    
    
    # for i in range(20):
    #     t = T.Compose([ToOpenCV(), RGBtoBGR()])

    #     ds = train_dss[0]

    #     frame, _ = ds[i]
    #     for j in range(args.consecutive_frames):
    #         cv2.imshow("output", t(frame[j]))
    #         cv2.waitKey(80)

    # cv2.destroyAllWindows()


    #TODO add crop to validation set.
    valid_dls = None
    if args.valid_model:
        if trf.transforms is not None and trf.crop:
            valid_crop = [trf_crop]
        else:
            valid_crop = None
        

        v_videos = [os.path.join(valid_dir, os.path.join(str(video), str(part))) 
                        for valid_dir in args.valid_dirs
                        for video, video_parts in zip(args.valid_videos, args.valid_parts) for part in video_parts]
    
        valid_videos = [video + ".pt" for video in v_videos]
        valid_angles = [video + ".txt" for video in v_videos]

        if args.dataset == "videods":
            valid_dss = [get_consecutive_frames_ds(video_path, angles_path, args.consecutive_frames, args.skips, valid_crop) 
                            for video_path, angles_path in zip(valid_videos, valid_angles)]
        elif args.dataset == "diskvideods":
            valid_dss = [get_disk_consecutive_frames_ds(video_path, args.consecutive_frames, args.skips, valid_crop) 
                            for video_path  in v_videos]

        valid_dls = [
            DataLoader(valid_ds, 32,
            num_workers=args.valid_workers, shuffle=False, pin_memory=True, persistent_workers=False) 
            for valid_ds in valid_dss]

    img_size = img_size if trf.crop_size is None else tuple(trf.crop_size)
    calib_params = CalibParams(img_size, args.consecutive_frames, args.lstm_hidden_size, args.lstm_num_layers, 
                                args.lstm_dropout, args.linear_layers, args.linear_dropout)
    model = CalibModel(calib_params)
    model.to(dev)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    
    print(f"CNN SHAPE OUT: {model.cnn_shape_out}")
    print(f"Number of parameters {pytorch_total_params}")
    print(f"MEMORY USED BY THE MODEL IN BYTES {mem}")
    print(f"Frame shape {train_dss[0][0][0].shape}")
    if args.valid_model:
        print(f"Valid shape {valid_dss[0][0][0].shape}")

    
    if args.opt == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)


    if args.loss == "mse":
        loss = nn.MSELoss()
    elif args.loss == "rmse":
        loss = RMSELoss()


    scheduler =  torch.optim.lr_scheduler.MultiStepLR(opt, args.scheduler_epochs, gamma=args.scheduler_gamma)

    history = History(args.training_info_dir, args.train_videos, args.valid_videos, 
                        model, opt, loss, scheduler, args.batch_size, args.history_active)
    
    grad_flow = GradientFlowVisualization(args.training_info_dir, args.grad_flow_epochs, args.grad_flow_active)
    
    activation_map = ActivationMapVisualization(args.training_info_dir, model, dev, args.activation_map_active)
    
    trn.fit(args.epochs, history, grad_flow, train_dls, valid_dls, dev, verbose=args.verbose)

    #TODO: delete dataloaders here? 

    if args.history_save_train_info:
        history.save_training_info()
    
    if args.history_save_model:
        history.save_model()


    #TODO load the test set already cropped
    if args.test_model:
        #load test data here
        test_transform = None
        if trf.crop:
            test_transform = [CropVideo(*list(trf.crop_args))]

        video_test_path = os.path.join(args.test_dir, str(args.test_video))
        test_ds = ConsecutiveFramesDataset(os.path.join(video_test_path, "test_video.pt"),
                    os.path.join(video_test_path, "angles.txt"), args.consecutive_frames, args.skips,
                    test_transform)
        
        test_model = TestModel(args.training_info_dir, os.path.join(video_test_path, "angles.txt"), 
                        test_ds, model)
            
        test_model.test()
    

    if activation_map:
        timer.start()
        for layer_name in args.activation_map_layers:
            activation_map.register_activation_map(layer_name)

        #NOTE: TECHNICALLY THE VISUALIZATION OF THE ACTIVATION MAPS SHOULD BE DONE ON VALID/TEST SET. FOR NOW I'M GOING TO DO IT 
        # ON THE TRAINING SET.
        act_map_dss = [valid_dss[i] for i in args.activation_map_valid_ds]
        activation_map_frames = list(args.activation_map_valid_frames)
        if args.activation_map_test:
            act_map_dss += [test_ds]
            activation_map_frames += [list(args.activation_map_test_frames)]

        activation_map.trigger_activation_maps(act_map_dss, activation_map_frames)
        activation_map.save_activation_maps()
        timer.end()
        print(f"Activation map: {timer}")

if __name__ == "__main__":
    main()