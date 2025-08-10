import os 
import traceback
from typing import Dict, Any, Optional, List, Tuple
import sys
from pathlib import Path
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from Managers.ConfigManager import ConfigManager
from Modules.CustomLogger import CustomLogger
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import cv2
import numpy as np
import PIL
import numpy as np
import time
import re
import json
from glob import glob

class VLMProcessor:
    def __init__(self, name: str = "vlm_processor") -> None:
        """
        Initialize the VLMProcessor with configuration settings.
        
        Args:
            name (str): Name of the VLM processor instance.
        """
        try:
            self.name: str = name
            self.config_manager: ConfigManager = ConfigManager.get_instance()
            custom_logger: CustomLogger = CustomLogger(self.name)
            self.logger = custom_logger.get_logger(
                log_file=f"logs/{name}.log",
                log_level=self.config_manager.get("LOG_LEVEL"),
                log_to_console=self.config_manager.get("LOG_TO_CONSOLE"),
            )
            self.model_name = "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit"
            self.running: bool = True
            self.PROCESSSED_IMAGE_FOLDER = self.config_manager.get("PROCESSSED_IMAGE_FOLDER", "Processed_Data")
            if not os.path.exists(self.PROCESSSED_IMAGE_FOLDER):
                os.makedirs(self.PROCESSSED_IMAGE_FOLDER)

            # default: Load the model on the available device(s)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype="auto", device_map="auto"
            )
            self.min_pixels = 256*28*28
            self.max_pixels = 1024*28*28
            self.BACK_IMAGE_FOLDER = self.config_manager.get("BACK_IMAGE_FOLDER", "Back_Images")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = AutoProcessor.from_pretrained(self.model_name, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
            self.prompt ="""Act like a highly accurate optical character recognition (OCR) model specialized in extracting container logistics metadata. Your task is to analyze the provided image or text content and return all relevant shipping container metadata in a well-structured JSON format.

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
            # self.logger.info(f"VLMProcessor {self.name} initialized with settings: {self.config_manager.defaults}")
            
            
        except Exception as e:
            print(f"Error initializing VLMProcessor: {e} | {traceback.format_exc()}")
    
    def parse_and_save_output(self,output_text, filename="container_output.json",save_folder="Processed_Data"):
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
                file_path = os.path.join(save_folder, filename)
                with open(file_path, 'w') as f:
                    json.dump(parsed_json, f, indent=2)
                self.logger.info(f"JSON saved to {file_path}")
                return parsed_json
            except json.JSONDecodeError as e:
                # print(f"Error parsing JSON: {e}")
                self.logger.error(f"Error parsing JSON: {e}")
                return None
        else:
            # print("No JSON found in output")
            self.logger.warning("No JSON found in output")
            return None
        
    def  predict(self,image:np.ndarray):
        try:
            if image is None or not isinstance(image, np.ndarray):
                self.logger.error("Invalid image input. Please provide a valid image.")
                return None
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
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
            start_time = time.time()
            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            print(f"Generating the output")
            # Inference: Generation of the output
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            print(f"decoding")
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(f"Inference time : {time.time()-start_time}")
            time_taken = time.time() - start_time
            self.logger.info(f"Inference time: {time_taken:.2f} seconds")
            self.logger.info(f"Output: {output_text}")
            print(output_text)
            return output_text
                
            
          
            
            
        
        except Exception as e:
            self.logger.error(f"Error in predict method: {e} | {traceback.format_exc()}")
            return None
    
    def start(self):
        while self.running:
            try:
                images = glob(os.path.join(self.BACK_IMAGE_FOLDER, "*.jpg"))
                if len(images) == 0:
                    self.logger.info("No images found in the BACK_IMAGE_FOLDER.")
                    time.sleep(5)
                    continue
                for image_path in images:
                    self.logger.info(f"Processing image: {image_path}")
                    image = cv2.imread(image_path)
                    if image is not None:
                        output = self.predict(image)
                        image_name = os.path.basename(image_path).split('.')[0]
                        json_filename = f"{image_name}_output.json"
                        
                        if output:
                            image_save_path = os.path.join(self.PROCESSSED_IMAGE_FOLDER, image_name + ".jpg")
                            self.logger.info(f"Saving processed image to {image_save_path}")
                            cv2.imwrite(image_save_path, image)
                            self.parse_and_save_output(output, filename=json_filename, save_folder=self.PROCESSSED_IMAGE_FOLDER)
                    else:
                        self.logger.error(f"Failed to read image: {image_path}")
                
            except Exception as e:
                self.logger.error(f"Error in VLMProcessor loop: {e} | {traceback.format_exc()}")
                time.sleep(1)

if __name__ == "__main__":
    vlm_processor = VLMProcessor()
    # Example usage with a sample image path
    image_path = "C5.png"  # Replace with your actual image path
    image = cv2.imread(image_path)
    if image is not None:
        output = vlm_processor.predict(image)
        print("Output:", output)
    else:
        print("Failed to read the image.")