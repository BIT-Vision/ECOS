# Codes for "Continuous-time Object Segmentation using High Temporal Resolution Event Camera"

This repository contains the official codes for `Continuous-time Object Segmentation using High Temporal Resolution Event Camera`.
## Requirements
- torch >= 1.8.0
- torchvison >= 0.9.0
- ...

To installl requirements, run:
```bash
conda create -n ECOSNet python==3.7
pip install -r requirements.txt
```

## Data Organization
### EOS Dataset
Download the [EOS](https://drive.google.com/file/d/1qGugNiY3dMWgFwxYXurAJ4L_xjhq8SMQ/view?usp=share_link) dataset, then organize data as following format:
```
EventData
      |----00001
      |     |-----e2vid_images
      |     |-----event_5
      |     |-----event_image
      |     |-----event_label_format
      |     |-----event_ori
      |     |-----rgb_image
      |----00002
      |     |-----e2vid_images
      |     |-----event_5
      |     |-----event_image
      |     |-----event_label_format
      |     |-----event_ori
      |     |-----rgb_image
      |----...
      |----data.json
      |----event_train.txt
      |----event_test.txt
      |----event_camera_test.txt
      |----event_object_test.txt
```
Where `e2vid_images` contains the reconstruction image using [E2vid](https://github.com/uzh-rpg/rpg_e2vid), `event_5` contains the voxel with 5 bins, `event_image` contains event composition image, `event_label_format` contains the object mask, `event_ori` contains the original event stream, `rgb_image` contains the rgb modality images for each video.

### DAVIS_Event
This dataset is based on [DAVIS17](https://davischallenge.org/davis2017/code.html), we use [v2e](https://github.com/SensorsINI/v2e) to generate event stream.
Download the [DAVIS_Event](https://drive.google.com/file/d/1ydDTgAbiP18IU7jVU1JvFeJjJfQux4Df/view?usp=share_link) dataset, then organize data as following format:
```
davis_event
      |----bear
      |     |-----e2vid_images
      |     |-----event_5
      |     |-----event_image
      |     |-----event_label_format
      |     |-----event_ori
      |     |-----rgb_image
      |----bike-packing
      |     |-----e2vid_images
      |     |-----event_5
      |     |-----event_image
      |     |-----event_label_format
      |     |-----event_ori
      |     |-----rgb_image
      |----...
      |----data.json
      |----event_train.txt
      |----event_test.txt
```

## Training

### Training on EOS or DAVIS_Event dataset
Download encoder [checkpoint](https://drive.google.com/file/d/1HM1q0nWCWHdUcd8003u7w1LDfoxun0wH/view?usp=share_link). To train the ECOSNet on EOS or DAVIS_Event dataset, just modify the dataset root `$cfg.DATA.NEUROMORPHIC_ROOT` and `$cfg.TRAIN.ENCODER_PATH` in `config.py`, then run following command.
```bash
python train.py --gpu ${GPU-IDS} --exp_name ${experiment}
```
## Testing
Download the model pretrained on EOS dataset [checkpoint](https://drive.google.com/file/d/1AiLJFhvRvvHfQ_qFB2HcJpUcYjkyYNcg/view?usp=share_link) and on DAVIS_Event dataset [checkpoint](https://drive.google.com/file/d/1dhF0XvJQkA21bFBDoBnyrh_Z3uwdbIEL/view?usp=share_link).

To eval the ECOSNet network on (EOS Dataset or DAVIS_Event), modify `$cfg.DATA.NEUROMORPHIC_ROOT`, then run following command
```bash
python eval.py --checkpoint ${./checkpoint/EOS_best.pth.tar} --results ${./results/EOS}
```
The results will be saved as indexed png file at `${results}/`.

Additionally, you can modify some setting parameters in `config.py` to change configuration.

# Acknowledgement
This codebase is built upon [official TransVOS repository](https://github.com/sallymmx/TransVOS).
