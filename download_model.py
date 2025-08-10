import os 
import gdown
from Managers.ConfigManager import ConfigManager

file_url = 'https://drive.google.com/uc?id='
config_manager = ConfigManager.get_instance()
YOLO_MODEL_PATH = config_manager.get("YOLO_MODEL_PATH")

def check_and_download_model():
    try:
        if not os.path.exists(YOLO_MODEL_PATH):
            print("Downloading model")
            gdown.download(file_url+"1HMSIghBOi-uWPdOq13CugVUZBS9tXcnt",YOLO_MODEL_PATH,quiet=False)
        return True
    
    except Exception as e:
        print(f"Error checking or downloading model: {e}")
        return False

if __name__ == "__main__":
    check_and_download_model()
    print(f"Model {YOLO_MODEL_PATH} is ready for use.")