
import tempfile, shutil, os
from PIL import Image
import numpy as np

def save_upload_to_tempfile(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    uploaded_file.seek(0)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name

def render_image_preview(path):
    
    ext = path.lower()
    if ext.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(path)
        st = __import__("streamlit")
        st.image(img, use_container_width=True)
        return

    try:
        import pydicom
        ds = pydicom.dcmread(path)
        if hasattr(ds, 'pixel_array'):
            arr = ds.pixel_array
            st = __import__("streamlit")
            st.image(arr, use_container_width=True)
            return
    except Exception:
        pass

    try:
        import nibabel as nib
        import numpy as np
        img = nib.load(path)
        data = img.get_fdata()
        slice = data[..., data.shape[-1]//2]

        sl = slice - slice.min()
        sl = (sl / (sl.max() + 1e-9) * 255).astype(np.uint8)
        st = __import__("streamlit")
        st.image(sl, use_container_width=True)
        return
    except Exception:
        pass
    raise RuntimeError("No preview available for this file type or required libs not installed.")
