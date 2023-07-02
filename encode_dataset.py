import os
import sys
import yaml
import torch
sys.path.append("./model")
from pathlib import Path
from space_encoder import SpaceEncoder

def load_model():
    # -- load parameters which the model was trained with
    parameters  = yaml.safe_load(Path("./model/space_train_config.yaml").read_text())
    channel_size = parameters["MODEL_PARAM"]["CHANNEL_SIZE"]
    latent_dim = parameters["MODEL_PARAM"]["LATENT_DIM"]
    # -- initialize instance of model
    space_encoder = SpaceEncoder(channel_size, latent_dim)
    # -- load weights
    space_encoder.load_state_dict(torch.load("./model/weights/RomNetEncoder.pth"))
    space_encoder.eval()
    return space_encoder 

# -- normalization constant: the variables that were just to normalize full space
norm_param = {"velocity_x": {"max": 3.5961, "min": -1.5560},
              "velocity_y": {"max": 1.6762, "min": -1.6739},
              "temperature": {"max": 900, "min": 0},
              }

def normalize_dataset(dataset):
    # -- termperature
    t_max = norm_param["temperature"]["max"]
    t_min = norm_param["temperature"]["min"]
    dataset[:,:,:,2] = (dataset[:,:,:,2] - t_min) / (t_max-t_min)
    # -- velocity_x
    u_max = norm_param["velocity_x"]["max"]
    u_min = norm_param["velocity_x"]["min"]
    dataset[:,:,:,0] = (dataset[:,:,:,0] - u_min) / (u_max-u_min)
    # -- velocity_y
    v_max = norm_param["velocity_y"]["max"]
    v_min = norm_param["velocity_y"]["min"]
    dataset[:,:,:,1] = (dataset[:,:,:,1] - v_min) / (v_max-v_min)
    return dataset


if __name__=="__main__":
    # -- get arguments
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    # -- load model
    space_encoder = load_model()
    # -- extract videos
    for file in os.listdir(input_folder):
        if file.startswith("data_"):
            # -- load data file
            file_path = os.path.join(input_folder, file)
            full_space_data = torch.load(file_path).squeeze(0)[:, :, :, [2, 3, 4]]
            print(full_space_data.shape)
            # -- normalize dataset
            norm_data = normalize_dataset(full_space_data)
            # -- run inference
            latent_data = space_encoder(norm_data)
            # -- save in output dir
            torch.save(latent_data, os.path.join(output_folder, "latent_" + file.split("_")[1]))

