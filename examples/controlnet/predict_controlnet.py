from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch, json, os, re, math

def load_predict_model(base_model_path, controlnet_path):

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.safety_checker = lambda images, clip_input: (images, None)

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
        predict_dict["prompt"] = data["text1"]
        predict_dict["conditioning_image"] = os.path.join(predict_folder, data["conditioning_image1"])
        predict_data_dict.append(predict_dict)

    return predict_data_dict
def angle_calculation(angle, step):
    radians = math.radians(angle)
    results = []
    for i in range(step):
        n = i + 1
        m = math.cos(radians + 2 * n * math.pi / step)
        results.append(m)
    return results
def angle_encoder(batch_data):
    """
    use cos to encode the angle information
    input data: shape:(batch_size, data_shape)
    output_data: shape:(batch_size, num_anglaccelerate launch train_controlnet_loss_alos.py --pretrained_model_name_or_path="/nfsv4/23062676g/Network_python/diffusers/examples/controlnet/stable-diffusion-v1-5" --output_dir=models09111023 --dataset_name="/nfsv4/23062676g/Lunar_Dataset/alos_dem_sd" --resolution=224 --learning_rate=1e-6 --validation_image "/nfsv4/23062676g/Lunar_Dataset/NAC_sd_dataset/train/M1302615397RE_B_25.png"  --validation_prompt "The sun elevation angle is 55.86 sun azimuth angle is 270.23"  --train_batch_size=20 --num_train_epochs=50
es.shape)
    """
    data_encoding = []
    for angle in batch_data:
        angle_encoding = angle_calculation(angle, 768)
        data_encoding.append(angle_encoding)


    return data_encoding

def extract_angle(angle_text):
    numbers = re.findall(r'-?\d+\.\d+', angle_text)
    numbers = [float(num) for num in numbers]
    return numbers

def predict_img(pipe, img_path, prompt):

    control_image = load_image(img_path)
    # generate image
    generator = torch.manual_seed(0)

    inputs_ids = extract_angle(prompt)

    prompt_input = torch.tensor([angle_encoder(inputs_ids)])


    image = pipe(
        num_inference_steps=20, generator=generator, image=control_image, prompt_embeds=prompt_input
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
    base_model_path = "/nfsv4/23062676g/Network_python/diffusers/examples/controlnet/stable-diffusion-v1-5"
    controlnet_model_path = "/nfsv4/23062676g/Network_python/diffusers/examples/controlnet/models_train10141027"
    json_file = "/nfsv4/23062676g/Lunar_Dataset/NAC_sd_dataset_loss_light_normalise_filter/val.json"
    predict_folder = "/nfsv4/23062676g/Lunar_Dataset/NAC_sd_dataset_loss_light_normalise_filter"
    save_folder = os.path.join(controlnet_model_path, 'predict' + os.path.basename(controlnet_model_path)[6:] + '_' +
                               os.path.basename(predict_folder)[-1:] + '_20')

    os.makedirs(save_folder, exist_ok=True)
    predict_app(base_model_path, controlnet_model_path, json_file, predict_folder, save_folder)