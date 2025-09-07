import os
import zipfile
import tempfile
import shutil
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pydicom
import tensorflow as tf
from keras.models import load_model

# # optional helper from your project; wrapper will continue if not present
# try:
#     from src.core.ai_insights import generate_insight
# except Exception:
#     generate_insight = None


# tf.get_logger().setLevel("ERROR")


class PETWrapper:
    """
    PET model wrapper to be used from src.services.inference.run_inference.
    - model_path: path to a Keras .h5 file OR a SavedModel directory (default tries
      'src/models/imaging/PET/model').
    - target_shape: (H, W) used to resize slices before per-slice inference.
    - suv_threshold: simple threshold used for lesion voxel counting.
    - custom_objects: passed to keras.load_model if needed for custom layers.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        target_shape: Tuple[int, int] = (128, 128),
        suv_threshold: float = 0.5,
        custom_objects: Optional[dict] = None,
    ):
        self.model_path = model_path or "src/models/imaging/PET/models/pet_classifier.h5"
        self.device = device
        self.target_shape = target_shape
        self.suv_threshold = suv_threshold
        self.custom_objects = custom_objects
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"PET model not found at {self.model_path}")
        # load Keras model (works for SavedModel dir or .h5)
        return load_model(self.model_path, custom_objects=self.custom_objects)

    def _extract_zip_and_find_dcm_dir(self, zip_path: str) -> Tuple[str, str]:
        tmpdir = tempfile.mkdtemp(prefix="pet_upload_")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        # pick directory with the most .dcm files
        best_dir = None
        max_count = 0
        for root, _, files in os.walk(tmpdir):
            count = sum(1 for f in files if f.lower().endswith(".dcm"))
            if count > max_count:
                max_count = count
                best_dir = root

        if best_dir is None or max_count == 0:
            shutil.rmtree(tmpdir)
            raise ValueError("No DICOM (.dcm) files found inside the uploaded zip.")

        return best_dir, tmpdir

    def _gather_dicom_series(self, path: str) -> Tuple[np.ndarray, float, Optional[str]]:
        """
        Accepts:
          - path to a .zip containing DICOMs
          - path to a folder containing .dcm files (recursive)
          - path to a single .dcm file
        Returns:
          - volume: np.ndarray shape (slices, Horig, Worig) normalized to [0,1]
          - voxel_volume_ml: voxel volume in mL computed using PixelSpacing & SliceThickness where available
          - cleanup_tmpdir: path to tmpdir that should be removed by caller (or None)
        """
        tmpdir_to_cleanup = None

        if os.path.isfile(path) and path.lower().endswith(".zip"):
            dicom_dir, tmpdir_to_cleanup = self._extract_zip_and_find_dcm_dir(path)
        elif os.path.isdir(path):
            dicom_dir = path
        elif os.path.isfile(path) and path.lower().endswith(".dcm"):
            dicom_dir = os.path.dirname(path)
        else:
            raise ValueError("Unsupported file type for PET wrapper. Provide .zip (of .dcm) or folder or .dcm file.")

        dcm_files = []
        for root, _, files in os.walk(dicom_dir):
            for f in files:
                if f.lower().endswith(".dcm"):
                    dcm_files.append(os.path.join(root, f))

        if not dcm_files:
            if tmpdir_to_cleanup:
                shutil.rmtree(tmpdir_to_cleanup)
            raise ValueError("No .dcm files found in DICOM folder.")

        # read slices
        slices = [pydicom.dcmread(f) for f in dcm_files]
        # sort by InstanceNumber if available
        slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))

        images = np.stack([s.pixel_array for s in slices]).astype(np.float32)
        images = (images - images.min()) / (images.max() - images.min() + 1e-6)

        # try compute approximate voxel volume in mL (pixel spacing in mm, thickness in mm)
        try:
            px, py = slices[0].PixelSpacing
            thickness = getattr(slices[0], "SliceThickness", 1.0)
            voxel_volume_ml = (float(px) * float(py) * float(thickness)) / 1000.0
        except Exception:
            voxel_volume_ml = 1.0

        return images, voxel_volume_ml, tmpdir_to_cleanup

    def _preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        # resize each slice and ensure shape (N, H, W, 1)
        out = []
        for sl in volume:
            img = sl[..., np.newaxis]
            resized = tf.image.resize_with_pad(img, self.target_shape[0], self.target_shape[1]).numpy()
            out.append(resized)
        return np.array(out, dtype=np.float32)

    def _predict_slices(self, X: np.ndarray) -> np.ndarray:
        # per-slice inference (keeps memory usage small)
        preds = []
        for i in range(X.shape[0]):
            slice_img = np.expand_dims(X[i], axis=0)  # (1,H,W,1)
            p = self.model.predict(slice_img, verbose=0)
            p = np.asarray(p)
            # flatten to 1D for aggregation
            if p.ndim > 1:
                p = p[0]
            preds.append(p)
        return np.array(preds)

    def compute_pet_metrics(self, volume: np.ndarray, voxel_volume_ml: float, suv_threshold: Optional[float] = None) -> Dict[str, float]:
        suv_threshold = suv_threshold if suv_threshold is not None else self.suv_threshold
        suvmax = float(volume.max())
        mask = volume > suv_threshold
        lesion_voxels = int(mask.sum())
        lesion_volume_ml = lesion_voxels * voxel_volume_ml
        return {"suvmax": suvmax, "lesion_ml": lesion_volume_ml}

    def predict(self, file_path: str) -> Dict[str, Any]:
     tmpdir_to_cleanup = None
     try:
         volume, voxel_volume_ml, tmpdir_to_cleanup = self._gather_dicom_series(file_path)
         X = self._preprocess_volume(volume)  
         preds = self._predict_slices(X)      

         mean_pred = preds.mean(axis=0)


         if np.size(mean_pred) == 1:
             prob = float(np.reshape(mean_pred, (-1,))[0])
             label = "Cancer" if prob > 0.5 else "Healthy"
             confidence = prob if prob > 0.5 else 1.0 - prob
         else:
             pred_class = int(np.argmax(mean_pred))
             confidence = float(np.max(mean_pred))
             label = f"Class {pred_class}"

         metrics = self.compute_pet_metrics(volume, voxel_volume_ml)

         result = {
             "label": label,
             "confidence": float(confidence),
             "heatmap": None,
             "suvmax": float(metrics["suvmax"]),
             "lesion_ml": float(metrics["lesion_ml"]),
             "ai_insight": "No insight available." 
         }

         return result 

     finally:
         if tmpdir_to_cleanup and os.path.isdir(tmpdir_to_cleanup):
             shutil.rmtree(tmpdir_to_cleanup, ignore_errors=True)

