import os
import sys
import yaml
import torch
sys.path.append("./model")
from pathlib import Path
from space_decoder import SpaceDecoder

def load_model():
    # -- load parameters which the model was trained with
    parameters  = yaml.safe_load(Path("./model/space_train_config.yaml").read_text())
    channel_size = parameters["MODEL_PARAM"]["CHANNEL_SIZE"]
    latent_dim = parameters["MODEL_PARAM"]["LATENT_DIM"]
    # -- initialize instance of model
    space_decoder = SpaceDecoder(channel_size, latent_dim)
    # -- load weights
    space_decoder.load_state_dict(torch.load("./model/weights/RomNetDecoder.pth"))
    space_decoder.eval()
    print(space_decoder)
    return space_decoder 

# -- normalization constant: the variables that were just to normalize full space
norm_param = {"velocity_x": {"max": 3.5961, "min": -1.5560},
              "velocity_y": {"max": 1.6762, "min": -1.6739},
              "temperature": {"max": 900, "min": 0},
              }

def normalize_dataset(dataset):
    # -- termperature
    t_max = norm_param["temperature"]["max"]
    t_min = norm_param["temperature"]["min"]
    dataset[:,:,:,2] = dataset[:,:,:,2] * (t_max-t_min) + t_min 
    # -- velocity_x
    u_max = norm_param["velocity_x"]["max"]
    u_min = norm_param["velocity_x"]["min"]
    dataset[:,:,:,0] = dataset[:,:,:,0] * (u_max-u_min) + u_min 
    # -- velocity_y
    v_max = norm_param["velocity_y"]["max"]
    v_min = norm_param["velocity_y"]["min"]
    dataset[:,:,:,1] = dataset[:,:,:,1] * (v_max-v_min) + v_min 
    return dataset


if __name__=="__main__":
    # -- get arguments
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    # -- load model
    space_decoder = load_model()
    # -- extract videos
    for file in os.listdir(input_folder):
        if file.startswith("latent_"):
            # -- load data file
            file_path = os.path.join(input_folder, file)
            latent_data = torch.load(file_path)
            print(latent_data.shape)
            # -- run inference
            full_space_data = space_decoder(latent_data)
            # -- normalize dataset
            #norm_data = normalize_dataset(full_space_data)
            # -- save in output dir
            torch.save(full_space_data, os.path.join(output_folder, "recovered_" + file.split("_")[1]))
            

