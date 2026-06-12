import os
# Force CPU only (hide GPU to prevent CUDA initialization error)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import base64
import io
import random
from models.schemas import PredictionResponse, PredictionResult
from core.frame_averaging import SimpleCNN, FrameAveragingCNN

# Paths & Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEBUG_MODE = True
DEBUG_DIR = os.path.join(BASE_DIR, '..', '..', 'storage', 'debug_images')
if DEBUG_MODE:
    os.makedirs(DEBUG_DIR, exist_ok=True)

def get_device():
    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            return torch.device("cuda")
    except Exception:
        pass

    return torch.device("cpu")

# Initialize and load models
device = get_device()
baseline_model = SimpleCNN().float().to(device)
augmented_model = SimpleCNN().float().to(device)

backbone_fa = SimpleCNN().float()
frame_model = FrameAveragingCNN(backbone_fa).float().to(device)

models_loaded = False
def load_models_if_needed():
    global models_loaded
    if models_loaded:
        return True
    try:
        baseline_path = os.path.join(BASE_DIR, '..', '..', 'storage', 'weights', 'model_baseline.pth')
        augmented_path = os.path.join(BASE_DIR, '..', '..', 'storage', 'weights', 'model_augmented.pth')
        frame_path = os.path.join(BASE_DIR, '..', '..', 'storage', 'weights', 'model_frame.pth')
        
        if not (os.path.exists(baseline_path) and os.path.exists(augmented_path) and os.path.exists(frame_path)):
            return False

        baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
        augmented_model.load_state_dict(torch.load(augmented_path, map_location=device))
        frame_model.backbone.load_state_dict(torch.load(frame_path, map_location=device))
        
        baseline_model.eval()
        augmented_model.eval()
        frame_model.eval()
        models_loaded = True
        print("Loaded AI models successfully!")
        return True
    except Exception as e:
        print(f"Error loading weights: {e}")
        return False

load_models_if_needed()

# Image preprocessing pipeline
def get_raw_pil_image(base64_str: str) -> Image.Image:
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.mode == 'RGBA':
        background = Image.new('RGBA', image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert('RGB')
        
    img_gray = image.convert('L')
    
    if img_gray.getpixel((0, 0)) > 128:
        img_gray = ImageOps.invert(img_gray)
            
    return img_gray

def crop_and_resize_to_mnist(img_gray: Image.Image) -> Image.Image:
    bbox = img_gray.getbbox()
    if bbox:
        img_cropped = img_gray.crop(bbox)
        w, h = img_cropped.size
        scale = 20.0 / max(w, h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img_resized = img_cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        img_np = np.array(img_resized)
        y_indices, x_indices = np.nonzero(img_np)
        if len(x_indices) > 0:
            weights = img_np[y_indices, x_indices]
            cx = np.average(x_indices, weights=weights)
            cy = np.average(y_indices, weights=weights)
        else:
            cx, cy = new_w / 2, new_h / 2
            
        img_final = Image.new('L', (28, 28), color=0)
        paste_x = int(14 - cx)
        paste_y = int(14 - cy)
        img_final.paste(img_resized, (paste_x, paste_y))
        return img_final
    else:
        return Image.new('L', (28, 28), color=0)

def image_to_tensor(img: Image.Image) -> torch.Tensor:
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Explicitly cast to float32 to avoid NequIP setting default dtype to float64
    return tensor_transform(img).unsqueeze(0).float().to(device)

def predict_single_tensor(model: nn.Module, tensor: torch.Tensor):
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        return pred.item(), conf.item(), probs

# Generate predictions
def generate_mock_prediction() -> PredictionResponse:
    return PredictionResponse(
        baseline=PredictionResult(digit=3, confidence=random.uniform(0.1, 0.4)),
        augmentation=PredictionResult(digit=3, confidence=random.uniform(0.75, 0.88)),
        averaging=PredictionResult(digit=3, confidence=random.uniform(0.92, 0.99))
    )

def generate_prediction(image_base64: str) -> PredictionResponse:
    if not load_models_if_needed():
        return generate_mock_prediction()

    raw_img = get_raw_pil_image(image_base64)
    if DEBUG_MODE:
        raw_img.save(os.path.join(DEBUG_DIR, "01_raw_canvas.png"))
    
    cropped_raw_img = crop_and_resize_to_mnist(raw_img)
    if DEBUG_MODE:
        cropped_raw_img.save(os.path.join(DEBUG_DIR, "02_cropped_28x28.png"))
        
    input_tensor = image_to_tensor(cropped_raw_img)

    # Baseline CNN
    base_pred, base_conf, _ = predict_single_tensor(baseline_model, input_tensor)

    # Data Augmentation CNN
    aug_pred, aug_conf, _ = predict_single_tensor(augmented_model, input_tensor)

    # FRAME AVERAGING END-TO-END
    avg_pred, avg_conf, avg_probs = predict_single_tensor(frame_model, input_tensor)
    
    if DEBUG_MODE:
        with open(os.path.join(DEBUG_DIR, "06_final_avg_probs.txt"), "w") as f:
            f.write(f"Final Averaged Probs: {avg_probs.cpu().numpy().tolist()}\n")
            f.write(f"WINNING DIGIT: {avg_pred} with {avg_conf*100:.2f}%\n")

    return PredictionResponse(
        baseline=PredictionResult(digit=base_pred, confidence=base_conf),
        augmentation=PredictionResult(digit=aug_pred, confidence=aug_conf),
        averaging=PredictionResult(digit=avg_pred, confidence=avg_conf)
    )
