# import torch
# from pathlib import Path
# from PIL import Image
# from torchvision import transforms
# from .models import UNet  # segmentation backbone
# from .classifier import StenosisClassifier  # hypothetical classifier module
# from .utils import fusion_predict, make_mask, clear_mask

# class DeepSAClassifierWrapper:
#     def __init__(self, device="cpu"):
#         self.device = torch.device(device)
#         # Load segmentation network
#         self.seg_model = UNet(1, 1, 32)
#         seg_ckpt = torch.load("src/models/imaging/DeepSA/weights/fscad_36249.ckpt", map_location=device)
#         seg_state = {k.replace('module.', ''): v for k, v in seg_ckpt['netE'].items()}
#         self.seg_model.load_state_dict(seg_state)
#         self.seg_model.to(device).eval()

#         # Load stenosis classifier
#         self.classifier = StenosisClassifier(num_classes=3)  # normal, mild, severe
#         clf_ckpt = torch.load("src/models/imaging/DeepSA/weights/xcad_4afe3.ckpt", map_location=device)
#         self.classifier.load_state_dict(clf_ckpt)
#         self.classifier.to(device).eval()

#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5]),
#         ])

#     def predict(self, file_path):
#         img = Image.open(file_path).convert("L")
#         img_tensor = self.transform(img).unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             # Segmentation for auxiliary visualization
#             seg_mask = self.seg_model(img_tensor)
#             # Classification
#             logits = self.classifier(img_tensor)
#             probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

#         classes = ["No Stenosis", "Mild", "Severe"]
#         label_idx = probs.argmax()
#         return {
#             "label": classes[label_idx],
#             "confidence": float(probs[label_idx]),
#             "heatmap": seg_mask.squeeze().cpu().numpy(),
#             "ai_insight": f"Detected {classes[label_idx]} stenosis with confidence {probs[label_idx]:.2f}"
#         }

# src/models/imaging/deepsa/wrapper.py
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms as T
import cv2
from skimage import morphology

# import the model & utilities from the repo
from .models import UNet
# utils.py in the repo defines fusion_predict, make_mask, clear_mask
try:
    from .utils import fusion_predict, make_mask, clear_mask
    _HAS_UTILS = True
except Exception:
    _HAS_UTILS = False

SIZE = 512

# transforms mirrored from demo.py
_tfmc1 = T.Compose([
    T.Resize(SIZE),
    T.Lambda(lambda img: img),  # tophat is in datasets.tophat in demo; we'll use utils if available
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

        # instantiate UNet exactly as demo.py
        self.netE = UNet(1, 1, 32, bilinear=True).to(self.device)

        # load checkpoint: demo loads checkpoint['netE'] and strips 'module.' prefix
        checkpoint = torch.load(str(self.ckpt_path), map_location="cpu")
        if "netE" in checkpoint:
            state_dict = checkpoint["netE"]
        else:
            # fallback: maybe the ckpt itself is a state_dict
            state_dict = checkpoint

        # strip "module." if present (common when saved from DataParallel)
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
        # convert to grayscale first, like demo
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

        # If the utils from repo exist, use fusion_predict which implements the improved pipeline
        if auto_thresh and _HAS_UTILS:
            multiangle = "Multiangle" in options
            pad = 50 if "Pad margin" in options else 0

            x1, x2 = self._preprocess_pil(pil_img)
            # fusion_predict signature from repo: fusion_predict(netE, ["none"], x1, ...)
            # We'll mimic demo.py call exactly
            _, out1 = fusion_predict(self.netE, ["none"], x1, multiangle=multiangle, denoise=4, size=SIZE,
                                     cutoff=0.4, pad=pad, netE=True)
            _, out2 = fusion_predict(self.netE, ["none"], x2, multiangle=False, denoise=4, size=SIZE,
                                     cutoff=0.4, pad=pad, netE=True)
            # combine outputs similar to demo
            out1_np = np.array(out1)
            out2_np = np.array(out2)
            merged = np.expand_dims(np.max(np.concatenate((out1_np, out2_np), axis=2), axis=2), 2)
            out_merge = Image.fromarray(merged.repeat(3, 2).astype(np.uint8))
            mask_merge = make_mask(out_merge, remove_size=2000, local_kernel=21, hole_max_size=100)
            seg_img = clear_mask(mask_merge)

            # create overlay heatmap: overlay mask on original resized image
            seg_arr = np.array(seg_img.convert("L"))
            # compute confidence as fraction of positive pixels
            confidence = float((seg_arr > 0).sum()) / seg_arr.size
            overlay = _make_overlay(pil_img, seg_arr)
            label = "Vessel segmentation"
        else:
            # simple forward pass flow (auto_thresh=False or utils absent)
            pil_img_l = pil_img.convert("L")
            x1 = _tfmc1(pil_img_l)
            input_tensor = x1.unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred_y = self.netE(input_tensor)
            # pred_y shape: (1,1,H,W). demo uses sign then morphology
            pred_np = pred_y.cpu().numpy()
            seg_bool = (np.sign(pred_np) + 1) / 2  # convert -1/1 to 0/1
            seg_bool = seg_bool.astype(bool)
            # remove small objects
            seg_clean = morphology.remove_small_objects(seg_bool[0, 0], 500)
            seg_img = (seg_clean * 255).astype("uint8")
            # convert to PIL
            seg_pil = Image.fromarray(seg_img)
            confidence = float(seg_clean.sum()) / seg_clean.size
            overlay = _make_overlay(pil_img, seg_img)
            label = "Vessel segmentation"

        return {
            "label": label,
            "confidence": float(confidence),
            "heatmap": overlay,            # PIL.Image (suitable for st.image)
            "ai_insight": f"Detected vessel regions (area fraction: {confidence:.4f})",
            "meta": {"ckpt": str(self.ckpt_path)}
        }


def _make_overlay(orig_pil: Image.Image, mask_arr: np.ndarray):
    """Create an RGBA overlay: red mask over original (resized to mask size)."""
    # ensure mask is 2D uint8
    if mask_arr.dtype != np.uint8:
        mask = (mask_arr > 0).astype(np.uint8) * 255
    else:
        mask = mask_arr
    # resize original to match mask
    orig_resized = orig_pil.resize((mask.shape[1], mask.shape[0])).convert("RGBA")
    red_mask = Image.new("RGBA", orig_resized.size, (255, 0, 0, 0))
    alpha = Image.fromarray((mask > 0).astype("uint8") * 180).convert("L")
    red_mask.putalpha(alpha)
    overlay = Image.alpha_composite(orig_resized, red_mask)
    return overlay
