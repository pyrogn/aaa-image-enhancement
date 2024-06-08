import warnings

import numpy as np
import torch
from IAT_enhance.model.IAT_main import IAT
from PIL import Image
from torchvision.transforms import Normalize


def inference_iat(
    file_name: str, task: str = "enhance", normalize: bool = False, device: str = "cuda"
) -> np.ndarray:
    # Weights path
    exposure_pretrain = "../IAT_enhance/best_Epoch_exposure.pth"
    enhance_pretrain = "../IAT_enhance/best_Epoch_lol_v1.pth"

    normalize_process = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # Load Pre-train Weights
    model = IAT().to(device)
    if task == "exposure":
        model.load_state_dict(
            torch.load(exposure_pretrain, map_location=torch.device(device))
        )
    elif task == "enhance":
        model.load_state_dict(
            torch.load(enhance_pretrain, map_location=torch.device(device))
        )
    else:
        warnings.warn("Only could be exposure or enhance")
    model.eval()

    # Load Image
    img = Image.open(file_name)
    img = np.asarray(img) / 255.0
    if img.shape[2] == 4:
        img = img[:, :, :3]
    input_tensor = torch.from_numpy(img).float().to(device)
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
    if normalize:
        input_tensor = normalize_process(input_tensor)

    # Forward Network
    with torch.no_grad():
        _, _, enhanced_img = model(input_tensor)

    enhanced_image_np = enhanced_img.squeeze().permute(1, 2, 0).cpu().numpy()

    enhanced_image_np = (enhanced_image_np * 255).astype(np.uint8)
    return enhanced_image_np


# # Convert tensor to PIL image and save if needed
# enhanced_image_pil = Image.fromarray(enhanced_image_np)
# enhanced_image_pil.save('enhanced_image.jpg')
