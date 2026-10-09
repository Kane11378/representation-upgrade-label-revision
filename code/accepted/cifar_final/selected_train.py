"""Select TRAIN image rows from opaque pickle storage; never open CIFAR TEST.

The archive TRAIN member's image bytes remain opaque until take(ids). No full
50000-image NumPy array is reconstructed, and no benchmark label is returned.
Only sentinel IDs are decoded before the four-case DINO gate passes.
"""
import codecs
import io
import pickle
import tarfile
from pathlib import Path
import numpy as np

class OpaqueImageStorage:
    def __setstate__(self, state):
        version, shape, dtype, fortran, raw = state
        assert version == 1 and tuple(shape) == (50000, 3072)
        assert np.dtype(dtype) == np.dtype('uint8') and not fortran
        assert isinstance(raw, bytes) and len(raw) == 50000*3072
        self.raw = raw

def opaque_reconstruct(*args):
    return OpaqueImageStorage()

class ImageStorageUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name == '_reconstruct':
            return opaque_reconstruct
        if (module, name) == ('numpy', 'ndarray'): return OpaqueImageStorage
        if (module, name) == ('numpy', 'dtype'): return np.dtype
        if (module, name) == ('_codecs', 'encode'): return codecs.encode
        raise pickle.UnpicklingError(f'Unexpected TRAIN pickle global {module}.{name}')

class SelectedTrainImages:
    def __init__(self, path):
        with tarfile.open(Path(path), 'r:gz') as archive:
            member = archive.getmember('cifar-100-python/train')
            with archive.extractfile(member) as f: raw = f.read()
        d = ImageStorageUnpickler(io.BytesIO(raw), encoding='bytes').load()
        storage = d.get(b'data', d.get('data'))
        assert isinstance(storage, OpaqueImageStorage)
        # Metadata labels exist in the original member; discard, never expose.
        self.storage = storage
        del d, raw
        self.accessed_ids = []

    def take(self, ids):
        ids = np.asarray(ids)
        assert ids.ndim == 1 and ids.dtype.kind in 'iu'
        assert np.all((ids >= 0) & (ids < 50000))
        images = np.empty((len(ids), 32, 32, 3), dtype=np.uint8)
        for i, row in enumerate(ids.tolist()):
            b = self.storage.raw[row*3072:(row+1)*3072]
            images[i] = np.frombuffer(b, dtype=np.uint8).reshape(3,32,32).transpose(1,2,0)
        self.accessed_ids.extend(ids.tolist())
        return images

