import os
import cv2
import time
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import PIL
import re
import json
model_name = "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit"
device = "cuda" if torch.cuda.is_available() else "cpu"
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map= device
)
# try these
    # unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit #Done, no gpu ran available
    # unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit # Done working within 2 seconds, with min_pixel = 256*28*28, amx_pixel = 640*28*28
    # Qwen/Qwen2.5-VL-3B-Instruct
    # unsloth/gemma-3-12b-it-unsloth-bnb-4bit

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.

min_pixels = 256*28*28
max_pixels = 1024*28*28
processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
prompt = """Act like a highly accurate optical character recognition (OCR) model specialized in extracting container logistics metadata. Your task is to analyze the provided image or text content and return all relevant shipping container metadata in a well-structured JSON format.

Your output must follow these guidelines:
- Ensure all values are precise and relevant.
- Return only actual, confidently extracted information.
- Use null for any fields that are not found or unclear.

Extract and return the following fields:
1. "container_number": A unique identifier of the shipping container, typically four letters followed by seven digits (e.g., ABCD1234567).
2. "iso_code": The ISO 6346 classification code (e.g., 22G1, 45R1) found on the container.
3. "check_digit": A single digit specified just after the container number.
4. "manufacturer": If visible, include the manufacturer or brand name of the container.
5. "other_tags": Capture any additional shipping-related identifiers found (e.g., max_gross, tare, net, cu_cap , size, type, or owner code) as a dictionary of key-value pairs.

Ensure the final response is formatted like this:
{
  "container_number": "ABCD1234567",
  "iso_code": "22G1",
  "check_digit": "7",
  "manufacturer": "Maersk",
  "other_tags": {
    "owner_code": "ABCD",
    "max_gross": "24,600 KG / 57,100 LBS",
    "tare": "2,240 KG / 4,940 LBS",
    "net": "36,240 KG / 80,260 LBS",
    "cu_cap": "33.7 CUM / 117.7 CF"
  }
}
"""


def parse_and_save_output(output_text, filename="container_output.json"):
    """
    Parse the VLM output and save as JSON file
    """
    # Extract JSON from the markdown-wrapped output
    json_match = re.search(r'```json\n(.*?)\n```', output_text[0], re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            # Parse the JSON to validate it
            parsed_json = json.loads(json_str)
            # Save to file
            with open(filename, 'w') as f:
                json.dump(parsed_json, f, indent=2)
            print(f"JSON saved to {filename}")
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return None
    else:
        print("No JSON found in output")
        return None

def get_caption(image_path):
    image = cv2.imread(image_path)
    #convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the image to PIL format
    image = PIL.Image.fromarray(image)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    "image":image
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    start_time = time.time()
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    print(f"Generating the output")
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    print(f"decoding")
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f"Inference time : {time.time()-start_time}")
    print(output_text)
    return output_text

# folder_path = "raw_images/"
# all_images = os.listdir(folder_path)
all_images = ['Processed_Images/20250810_200933_1.jpg']
for image_path in all_images:
    # print(image_path)
    # complete_image_path = os.path.join(folder_path,image_path)
    complete_image_path = image_path
    # print(complete_image_path)
    caption = get_caption(complete_image_path)
    parse_and_save_output(caption, filename="container_output.json")
    print("Caption:")
    print(caption)
    # img = cv2.imread(complete_image_path)
    # folder_name = "inference"
    # os.makedirs(folder_name,exist_ok=True)
    # image_name = image_path.split("/")[-1].split(".")[0]
    # image_path = f'{folder_name}/{image_name}'
    # os.makedirs(f'{image_path}',exist_ok=True)
    # cv2.imwrite(f'{image_path}/{image_name}.jpg',img)
    # with open(f"{image_path}/output.txt","w+") as f:
    #     f.write(str(caption))
        # f.write("\n")
        # f.write(keywords)