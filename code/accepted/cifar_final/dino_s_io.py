"""Local, frozen DINO-S I/O; no downloader, test reader, or noisy-label reader.

The preprocess and encode/sync/close method ASTs are copied from the historical
successful source_io.py. Model construction is the historical local-loader
override, restricted to S/14. No method performs training.
"""
from __future__ import annotations
import hashlib, io, pickle, tarfile, time
from pathlib import Path
import numpy as np
from PIL import Image

TRAIN_MD5 = '16019d7e3df5f24257cddd939b257f8d'
TRAIN_SHA256 = '735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b'
TRAIN_TAR_SHA256 = '85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7'
CLEAN_LABEL_RAW_SHA256 = '8bf4935e7c77a3096270043c6f3b221b6ecb9c63d1a08608e9af66cff17e977a'
DINO_S_SHA256 = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


class NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype,
          ('numpy.core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,
          ('numpy._core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,
          ('numpy.core.multiarray','scalar'):np._core.multiarray.scalar,
          ('numpy._core.multiarray','scalar'):np._core.multiarray.scalar,
          ('_codecs','encode'):__import__('codecs').encode}
        if (module,name) not in allowed:raise pickle.UnpicklingError(f'Not allowed: {module}.{name}')
        return allowed[module,name]


def read_train(archive):
    """Read only the already present official TRAIN member, never TEST."""
    archive=Path(archive)
    if digest(archive)!=TRAIN_TAR_SHA256:raise ValueError('Official TRAIN archive SHA256 mismatch')
    with tarfile.open(archive,'r:gz') as t:
        member=t.getmember('cifar-100-python/train')
        if not member.isfile():raise ValueError('Invalid TRAIN member')
        raw=t.extractfile(member).read()
    if hashlib.md5(raw).hexdigest()!=TRAIN_MD5:raise ValueError('Official TRAIN MD5 mismatch')
    if hashlib.sha256(raw).hexdigest()!=TRAIN_SHA256:raise ValueError('Official TRAIN SHA256 mismatch')
    obj=NumpyOnlyUnpickler(io.BytesIO(raw),encoding='latin1').load()
    data=np.asarray(obj['data']);clean=np.asarray(obj['fine_labels'],np.int64)
    if data.shape!=(50000,3072) or data.dtype!=np.uint8 or clean.shape!=(50000,):
        raise ValueError('Unexpected CIFAR-100 TRAIN layout')
    if hashlib.sha256(clean.tobytes(order='C')).hexdigest()!=CLEAN_LABEL_RAW_SHA256:
        raise ValueError('Official clean fine-label identity mismatch')
    images=data.reshape(-1,3,32,32).transpose(0,2,3,1).copy()
    manifest=dict(image_source=str(archive)+'::cifar-100-python/train',
        archive_sha256=TRAIN_TAR_SHA256,train_md5=TRAIN_MD5,train_sha256=TRAIN_SHA256,
        clean_label_c_order_raw_bytes_sha256=CLEAN_LABEL_RAW_SHA256,
        rows=50000,official_test_deserialized=False,noisy_labels_read=False)
    return images,clean,manifest


def preprocess(images):
    import torch
    result=[]
    for x in images:
        im=Image.fromarray(x).resize((256,256),Image.Resampling.BICUBIC).crop((16,16,240,240))
        result.append(np.asarray(im,dtype=np.float32).transpose(2,0,1)/255.)
    a=np.stack(result)
    a=(a-np.array([.485,.456,.406],np.float32)[None,:,None,None])/np.array([.229,.224,.225],np.float32)[None,:,None,None]
    return torch.from_numpy(a)


class LocalDinoS:
    def __init__(self,code,weights,device,batch=64):
        import torch
        if digest(weights)!=DINO_S_SHA256:raise ValueError('DINO-S pinned weight SHA256 mismatch')
        self.torch=torch;self.device=torch.device(device);self.batch=batch;self.name='dinov2_vits14'
        # Local source + pretrained=False forbids the hub's network loader.
        self.model=torch.hub.load(str(Path(code)),self.name,source='local',pretrained=False)
        self.model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('DINO-S must remain frozen and in eval mode')
        if any(p.dtype!=torch.float32 for p in self.model.parameters()):
            raise ValueError('DINO-S must be FP32')
        self.params=sum(p.numel() for p in self.model.parameters())
        self.parameter_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters())
        self.weights=dict(path=str(Path(weights)),bytes=Path(weights).stat().st_size,sha256=digest(weights))
        self.warmed=False

    def sync(self):
        if self.device.type=='cuda':self.torch.cuda.synchronize(self.device)

    def encode(self,images):
        torch=self.torch;outs=[];prep=0.;forward=0.;transfer=0.
        wall=time.perf_counter()
        for i in range(0,len(images),self.batch):
            t=time.perf_counter();x=preprocess(images[i:i+self.batch]);prep+=time.perf_counter()-t
            self.sync();t=time.perf_counter();x=x.to(self.device);self.sync();transfer+=time.perf_counter()-t
            if not self.warmed:
                with torch.inference_mode():
                    for _ in range(3):self.model(x)
                self.sync();self.warmed=True
                # Warmup excluded from forward time, but retained in call wall time.
            self.sync();t=time.perf_counter()
            with torch.inference_mode():
                z=self.model(x).float();z=torch.nn.functional.normalize(z,dim=1)
            self.sync();forward+=time.perf_counter()-t
            t=time.perf_counter();out=z.cpu().numpy();self.sync();transfer+=time.perf_counter()-t;outs.append(out)
        return np.concatenate(outs),dict(rows=len(images),preprocess_seconds=prep,
            forward_seconds=forward,transfer_seconds=transfer,call_wall_seconds=time.perf_counter()-wall,
            measured_stage_seconds=prep+forward+transfer,name=self.name,batch=self.batch,
            device=str(self.device),parameter_bytes=self.parameter_bytes)

    def close(self):
        del self.model
        if self.device.type=='cuda':self.torch.cuda.empty_cache()
