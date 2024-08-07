from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch, json, os

def load_predict_model(base_model_path, controlnet_path):

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, torch_dtype=torch.float16
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed or when using Torch 2.0.
    pipe.enable_xformers_memory_efficient_attention()
    # memory optimization.
    pipe.enable_model_cpu_offload()

    return pipe

def load_json_file(json_file):
    data_list = []
    with open(json_file, 'r') as file:
        for line in file:
            # 解析每行 JSON 数据
            data = json.loads(line)
            data_list.append(data)

    return data_list

def load_predict_dataset(predict_folder, json_file):
    predict_data_dict = []
    dataset_list = load_json_file(json_file)
    for data in dataset_list:
        predict_dict = {}
        predict_dict["prompt"] = data["text"]
        predict_dict["conditioning_image"] = os.path.join(predict_folder, data["conditioning_image"])
        predict_data_dict.append(predict_dict)

    return predict_data_dict


def predict_img(pipe, img_path, prompt):

    control_image = load_image(img_path)
    # generate image
    generator = torch.manual_seed(0)
    image = pipe(
        prompt, num_inference_steps=20, generator=generator, image=control_image
    ).images[0]

    return image

def predict_app(base_model_path, controlnet_path, json_file, predict_folder, save_folder):

    pipe = load_predict_model(base_model_path, controlnet_path)
    predict_dataset = load_predict_dataset(predict_folder, json_file)

    for predict_data in predict_dataset:
        img_path = predict_data["conditioning_image"]
        prompt = predict_data["prompt"]
        predict_image = predict_img(pipe, img_path, prompt)
        save_name = os.path.join(save_folder, os.path.basename(img_path))
        predict_image.save(save_name)

if __name__ == "__main__":
    base_model_path = "runwayml/stable-diffusion-v1-5"
    controlnet_model_path = "/nfsv4/23062676g/Network_python/diffusers/examples/controlnet/models07301117"
    json_file = "/nfsv4/23062676g/Lunar_Dataset/NAC_sd_dataset2/val.json"
    predict_folder = "/nfsv4/23062676g/Lunar_Dataset/NAC_sd_dataset2"
    save_folder = "/nfsv4/23062676g/Network_python/diffusers/examples/controlnet/models07301117/predict"

    os.makedirs(save_folder, exist_ok=True)
    predict_app(base_model_path, controlnet_model_path, json_file, predict_folder, save_folder)