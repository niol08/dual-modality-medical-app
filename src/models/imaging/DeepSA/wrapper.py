
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms as T
import cv2
from skimage import morphology

from .models import UNet
try:
    from .utils import fusion_predict, make_mask, clear_mask
    _HAS_UTILS = True
except Exception:
    _HAS_UTILS = False

SIZE = 512


_tfmc1 = T.Compose([
    T.Resize(SIZE),
    T.Lambda(lambda img: img),  
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

_tfmc2 = T.Compose([
    T.Resize(SIZE),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])


class DeepSAWrapper:
    def __init__(self, ckpt_path: str = None, device: str = "cpu"):
        """
        ckpt_path: path to a checkpoint (the demo uses 'ckpt/fscad_36249.ckpt')
        device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.ckpt_path = Path(ckpt_path) if ckpt_path else (Path(__file__).parent / "ckpt" / "fscad_36249.ckpt")
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {self.ckpt_path}")

        self.netE = UNet(1, 1, 32, bilinear=True).to(self.device)

        checkpoint = torch.load(str(self.ckpt_path), map_location="cpu")
        if "netE" in checkpoint:
            state_dict = checkpoint["netE"]
        else:
            state_dict = checkpoint

        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.netE.load_state_dict(new_state_dict)
        self.netE.to(self.device)
        self.netE.eval()

    def _preprocess_pil(self, pil_img: Image.Image, use_tophat: bool = False):
        """
        Return transformed tensors (x1, x2) analogous to demo.py
        If the repo's datasets.tophat exists, fusion_predict will use it.
        Here we approximate behavior by using transforms; fusion_predict handles internals.
        """
        img_l = pil_img.convert("L")
        x1 = _tfmc1(img_l)
        x2 = _tfmc2(img_l)
        return x1, x2

    def predict(self, image_path: str, auto_thresh: bool = True, options: list = None):
        """
        Run inference on an image file path (PNG/JPG). Returns a dict:
        { label, confidence, heatmap (PIL.Image), ai_insight, meta }
        """
        options = options or []
        pil_img = Image.open(image_path)

        if auto_thresh and _HAS_UTILS:
            multiangle = "Multiangle" in options
            pad = 50 if "Pad margin" in options else 0

            x1, x2 = self._preprocess_pil(pil_img)
            _, out1 = fusion_predict(self.netE, ["none"], x1, multiangle=multiangle, denoise=4, size=SIZE,
                                     cutoff=0.4, pad=pad, netE=True)
            _, out2 = fusion_predict(self.netE, ["none"], x2, multiangle=False, denoise=4, size=SIZE,
                                     cutoff=0.4, pad=pad, netE=True)
            out1_np = np.array(out1)
            out2_np = np.array(out2)
            merged = np.expand_dims(np.max(np.concatenate((out1_np, out2_np), axis=2), axis=2), 2)
            out_merge = Image.fromarray(merged.repeat(3, 2).astype(np.uint8))
            mask_merge = make_mask(out_merge, remove_size=2000, local_kernel=21, hole_max_size=100)
            seg_img = clear_mask(mask_merge)

      
            seg_arr = np.array(seg_img.convert("L"))
            confidence = float((seg_arr > 0).sum()) / seg_arr.size
            overlay = _make_overlay(pil_img, seg_arr)
            label = "Vessel segmentation"
        else:
            pil_img_l = pil_img.convert("L")
            x1 = _tfmc1(pil_img_l)
            input_tensor = x1.unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred_y = self.netE(input_tensor)
            pred_np = pred_y.cpu().numpy()
            seg_bool = (np.sign(pred_np) + 1) / 2 
            seg_bool = seg_bool.astype(bool)
            seg_clean = morphology.remove_small_objects(seg_bool[0, 0], 500)
            seg_img = (seg_clean * 255).astype("uint8")
            seg_pil = Image.fromarray(seg_img)
            confidence = float(seg_clean.sum()) / seg_clean.size
            overlay = _make_overlay(pil_img, seg_img)
            label = "Vessel segmentation"

        return {
            "label": label,
            "confidence": float(confidence),
            "heatmap": overlay,            
            "ai_insight": f"Detected vessel regions (area fraction: {confidence:.4f})",
            "meta": {"ckpt": str(self.ckpt_path)}
        }


def _make_overlay(orig_pil: Image.Image, mask_arr: np.ndarray):
    """Create an RGBA overlay: red mask over original (resized to mask size)."""
    if mask_arr.dtype != np.uint8:
        mask = (mask_arr > 0).astype(np.uint8) * 255
    else:
        mask = mask_arr
    orig_resized = orig_pil.resize((mask.shape[1], mask.shape[0])).convert("RGBA")
    red_mask = Image.new("RGBA", orig_resized.size, (255, 0, 0, 0))
    alpha = Image.fromarray((mask > 0).astype("uint8") * 180).convert("L")
    red_mask.putalpha(alpha)
    overlay = Image.alpha_composite(orig_resized, red_mask)
    return overlay
