
from importlib import import_module

_MODEL_MAP = {
    # biosignals
    "EEG": "src.models.biosignal.eeg_model.EEGModel",
    "ERP": "src.models.biosignal.eeg_model.EEGModel",
    "EOG": "src.models.biosignal.eeg_model.EEGModel",
    "EMG": "src.models.biosignal.eeg_model.EEGModel",
    "ECG": "src.models.biosignal.eeg_model.EEGModel",
    # imaging
    "MRI": "src.models.imaging.mri_model.MRIModel",
    "CT": "src.models.imaging.mri_model.MRIModel",
    "X-ray": "src.models.imaging.mri_model.MRIModel",
    "PET": "src.models.imaging.mri_model.MRIModel",
     "Angiogram": "src.models.imaging.deepsa.wrapper.DeepSAWrapper",
}

_SINGLETONS = {}

def _import_class(path):
    module_name, class_name = path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, class_name)

def get_model_for_modality(modality):
    key = modality
    path = _MODEL_MAP.get(key)
    if path is None:
        return None
    if path not in _SINGLETONS:
        cls = _import_class(path)
        _SINGLETONS[path] = cls()
    return _SINGLETONS[path]
