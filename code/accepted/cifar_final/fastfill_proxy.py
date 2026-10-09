"""One fixed frozen-epoch79 FastFill forward on the locked4000 history only."""
from __future__ import annotations

import datetime
import hashlib
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "8"
sys.dont_write_bytecode = True
import numpy as np
from encode_reserve import ROOT, PROJECT, TASK, asset, guards, read, require, save_array, semantic, sha, verify, write_new

CHECKPOINT = PROJECT / "fastfill_dino_normalized_adaptation_02/model/FINAL_EPOCH_79.pt"
MODEL_SHA = "0ac692e1e9dd0994e89a3f40fbc165a34cd097503ded8ac56cea0249712538f8"
ARCHITECTURE_SHA = "a96712e7b04a3ba9d8f1622833338fd2b4da75c5a338224de0baec1d93f0b3a6"
ARCHITECTURE_BLOB = "30b37be58a3aa372b2792e238426740ec7ff6d6c"
DIMS = {"old_embedding_dim": 384, "new_embedding_dim": 768,
        "side_info_dim": 384, "inner_dim": 2048, "sigma_dim": 1}


def runtime():
    import torch
    expected = PROJECT / "capability_feasibility_01/.venv/Scripts/python.exe"
    require(os.path.normcase(os.path.realpath(sys.executable)) == os.path.normcase(os.path.realpath(expected)),
            "Existing frozen CUDA frontend interpreter required")
    require(torch.__version__ == "2.11.0+cu128" and np.__version__ == "2.5.2" and torch.version.cuda == "12.8",
            "Original frozen frontend Torch/NumPy/CUDA runtime differs")
    require(torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5090",
            "Original frozen frontend CUDA device unavailable")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch, {"executable": sys.executable, "torch": torch.__version__, "numpy": np.__version__,
                   "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "device": "cuda",
                   "threads": 8, "FP32": True, "matmul_precision": "highest", "TF32": False,
                   "deterministic": True, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "AMP": False,
                   "environment_installed_or_modified": False}


def main():
    marker = ROOT / "logs/FASTFILL_PROXY_ATTEMPT_STARTED.json"
    output = ROOT / "FASTFILL_PROXY_LOCK.json"
    require(not marker.exists() and not output.exists(), "Frozen FastFill proxy mapping already attempted; no automatic retry")
    public, selected, provenance = guards()
    cache_path = ROOT / "FEATURE_CACHE_LOCK.json"
    cache = read(cache_path)
    require(cache.get("status") == "FROZEN" and cache["rows"] == 6000, "Frozen reserve feature cache required")
    identity = read(ROOT / "DINO_IDENTITY_GATE.json")
    require(identity.get("status") == "PASS" and len(identity["cases"]) == 4 and all(row["pass"] for row in identity["cases"]),
            "All four historical DINO numerical cases must PASS before FastFill")
    require(cache["numerical_identity_gate"]["sha256"] == sha(ROOT / "DINO_IDENTITY_GATE.json"),
            "Feature cache numerical identity binding changed")
    for key in provenance:
        require(cache["provenance"][key]["sha256"] == provenance[key]["sha256"],
                "Feature cache protocol/source binding changed")
    global_ids = np.load(verify(cache["assets"]["global_ids"]), allow_pickle=False)
    require(global_ids.dtype == np.int64 and np.array_equal(global_ids, selected), "Feature cache ID order differs from frozen protocol")
    # Do not open B or any labels/truth. S is a read-only mmap; only history
    # rows are copied into actual CPU/GPU forward tensors.
    s_path = verify(cache["assets"]["S"])
    old = np.load(s_path, allow_pickle=False, mmap_mode="r")
    require(old.shape == (6000, 384) and old.dtype == np.float32, "S feature cache schema differs")
    rows = np.asarray(public["split"]["groups"]["history"]["rows"], dtype=np.int64)
    history_ids = selected[rows]
    require(rows.shape == (4000,) and np.array_equal(rows, np.arange(4000, dtype=np.int64)), "Locked history rows differ")
    require(CHECKPOINT.stat().st_size == 60189229 and sha(CHECKPOINT) == MODEL_SHA,
            "Task-fixed normalized epoch79 checkpoint differs")
    architecture_path = ROOT / "code/official_transformations.py"
    raw = architecture_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ARCHITECTURE_SHA
            and hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == ARCHITECTURE_BLOB,
            "Byte-identical official architecture source differs")
    torch, rt = runtime()
    write_new(marker, {"task": TASK, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "single_attempt": True, "code": asset(__file__), "provenance": provenance,
                       "feature_cache_lock": asset(cache_path), "history_rows": rows.tolist(),
                       "history_global_ids": history_ids.tolist(), "checkpoint": asset(CHECKPOINT),
                       "batch_size": 1024, "post_normalization": False, "pilot_bias": False,
                       "sigma_evidence_selection": False, "labels_or_eval_truth_read": False})
    spec = importlib.util.spec_from_file_location("fixed_final_unseen_official_fastfill_architecture", architecture_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    cpu_rng_before = torch.random.get_rng_state().clone()
    with torch.device("meta"):
        net = module.MLP_BN_SIDE_PROJECTION_SIGMA(**DIMS)
    net.load_state_dict(state, strict=True, assign=True)
    require(torch.equal(cpu_rng_before, torch.random.get_rng_state()), "Checkpoint materialization consumed CPU RNG")
    require(sum(p.numel() for p in net.parameters()) == 15026689 and not any(p.is_meta for p in net.parameters()),
            "Exact official FastFill architecture parameter identity differs")
    require(all(not value.is_floating_point() or value.dtype == torch.float32 for value in state.values()),
            "Frozen FastFill state must be FP32")
    net = net.to("cuda").eval().requires_grad_(False)
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state().clone()
    bn_before = {name: value.clone() for name, value in net.state_dict().items()
                 if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    parameter_count = sum(p.numel() for p in net.parameters())
    parameter_bytes = sum(p.numel() * p.element_size() for p in net.parameters())
    state_bytes = sum(value.numel() * value.element_size() for value in net.state_dict().values())
    proxies = []
    sigmas = []
    actual_batches = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rows), 1024):
            batch_rows = rows[start:start + 1024]
            chunk = np.ascontiguousarray(old[batch_rows], dtype=np.float32)
            require(np.isfinite(chunk).all(), "History S features are nonfinite")
            x = torch.from_numpy(chunk).to("cuda")
            x4 = x[:, :, None, None]
            values = net(x4, x4)
            require(values.shape == (len(chunk), 769, 1, 1) and bool(torch.isfinite(values).all()),
                    "Frozen FastFill interface schema/nonfinite output failure")
            proxies.append(values[:, :768, 0, 0].cpu().numpy())
            sigmas.append(values[:, 768, 0, 0].cpu().numpy())
            actual_batches.append(len(chunk))
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    require(actual_batches == [1024, 1024, 1024, 928], "Fixed history batch1024 recipe differs")
    require(not net.training and not any(p.requires_grad for p in net.parameters()), "Frozen eval/parameter boundary changed")
    require(all(torch.equal(before, net.state_dict()[name]) for name, before in bn_before.items()),
            "Frozen eval BatchNorm state changed")
    require(torch.equal(cpu_rng_before, torch.random.get_rng_state()) and torch.equal(cuda_rng_before, torch.cuda.get_rng_state()),
            "Frozen FastFill forward consumed RNG")
    proxy = np.concatenate(proxies)
    sigma = np.concatenate(sigmas)
    require(proxy.shape == (4000, 768) and sigma.shape == (4000,) and proxy.dtype == sigma.dtype == np.float32,
            "Frozen4000 proxy/sigma output layout differs")
    assets = {"proxy": save_array(ROOT / "proxy/proxy.npy", proxy), "sigma": save_array(ROOT / "proxy/sigma.npy", sigma)}
    require(sha(CHECKPOINT) == MODEL_SHA and sha(s_path) == cache["assets"]["S"]["sha256"],
            "Checkpoint or S source changed during frozen mapping")
    write_new(ROOT / "logs/FASTFILL_PROXY_ACCESS.json", {"task": TASK, "S_cache": cache["assets"]["S"],
              "history_rows": rows.tolist(), "history_global_ids": history_ids.tolist(), "forward_rows": 4000,
              "batch_rows": actual_batches, "B_feature_rows_read": 0, "labels_or_eval_truth_read": False,
              "eval_or_E_forward_rows": 0, "DINO_forward": False, "training": False,
              "sigma_saved_only": True, "sigma_evidence_selection": False, "RNG_selection": False})
    lock = {"task": TASK, "status": "FROZEN", "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "assets": assets, "checkpoint": asset(CHECKPOINT), "architecture_source": asset(architecture_path),
            "architecture": "MLP_BN_SIDE_PROJECTION_SIGMA", "dimensions": DIMS,
            "parameter_count": parameter_count, "parameter_bytes": parameter_bytes, "state_bytes": state_bytes,
            "history_rows": rows.tolist(), "history_global_ids": history_ids.tolist(),
            "history_ids_semantic": semantic(history_ids), "batch_size": 1024, "batch_rows": actual_batches,
            "eval": True, "eval_mode": True, "frozen": True, "BN_unchanged": True, "RNG_unchanged": True,
            "side_info": "same old S feature", "input_shape": "Nx384x1x1", "output_shape": "Nx769x1x1",
            "post_normalization": False, "pilot_bias": False, "sigma_evidence_selection": False, "sigma_selection": False,
            "sigma_saved_only": True, "labels_or_eval_truth_read": False, "B_features_read": False,
            "no_training_or_head_loading": True, "provenance": provenance, "feature_cache_lock": asset(cache_path),
            "runtime": rt, "cost": {"rows": 4000, "wall_seconds": wall, "includes_host_device_transfer": True,
                                     "includes_model_load": False, "model": "normalized FastFill frozen epoch79"}}
    write_new(output, lock)
    print(__import__("json").dumps({"status": "FASTFILL_PROXY_FROZEN", "assets": assets, "cost": lock["cost"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        failure = ROOT / "logs/FASTFILL_PROXY_FAILED_STOP.json"
        if not failure.exists():
            write_new(failure, {"task": TASK, "status": "FAILED_STOP", "error": str(error),
                               "traceback": traceback.format_exc(), "automatic_retry_or_method_change": False})
        raise
