from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from procrustes_alignment import fit_rectangular_procrustes, apply_procrustes, geometry_error

def main() -> None:
    rng = np.random.default_rng(20261005)
    checks = {}
    n, d_old, d_new = 700, 384, 768
    old = rng.normal(size=(n, d_old))
    q, _ = np.linalg.qr(rng.normal(size=(d_new, d_old)))
    true_r = q.T
    shift = rng.normal(scale=.2, size=d_new)
    new = shift + old @ true_r

    fit = fit_rectangular_procrustes(old[:512], new[:512], mode="isometry")
    pred = apply_procrustes(fit, old[512:])
    checks["exact_map_max_abs"] = float(np.max(np.abs(pred-new[512:])))
    checks["orthogonality_fro"] = geometry_error(fit)
    assert checks["exact_map_max_abs"] < 2e-12
    assert checks["orthogonality_fro"] < 2e-12

    probe = old[512:612]
    mapped = apply_procrustes(fit, probe)
    dx = probe[:,None,:]-probe[None,:,:]
    dz = mapped[:,None,:]-mapped[None,:,:]
    old_d2 = np.sum(dx*dx, axis=2)
    new_d2 = np.sum(dz*dz, axis=2)
    checks["pairwise_distance_sq_max_abs"] = float(np.max(np.abs(old_d2-new_d2)))
    assert checks["pairwise_distance_sq_max_abs"] < 3e-12

    scale_true=1.7
    noisy_new=shift+scale_true*old@true_r+rng.normal(scale=.03,size=(n,d_new))
    sim=fit_rectangular_procrustes(old[:512],noisy_new[:512],mode="similarity")
    checks["similarity_fitted_scale"]=sim.scale
    assert abs(sim.scale-scale_true)<.02
    mapped=apply_procrustes(sim,probe)
    dz=mapped[:,None,:]-mapped[None,:,:]
    new_d2=np.sum(dz*dz,axis=2)
    checks["scaled_pairwise_distance_sq_max_rel"]=float(
        np.max(np.abs(new_d2-sim.scale**2*old_d2)/np.maximum(1.,sim.scale**2*old_d2)))
    assert checks["scaled_pairwise_distance_sq_max_rel"] < 2e-14

    pilot=fit_rectangular_procrustes(old[:512],noisy_new[:512],mode="similarity",
        old_pilot=old[512:612],new_pilot=noisy_new[512:612])
    a=apply_procrustes(sim,old[612:662]); b=apply_procrustes(pilot,old[612:662])
    checks["pilot_bias_pairwise_invariance"]=float(np.max(np.abs((a-a[0])-(b-b[0]))))
    assert checks["pilot_bias_pairwise_invariance"] < 2e-14

    variable_old=np.vstack([np.zeros((1,d_old)),np.eye(3,d_old)])
    variable_pred=apply_procrustes(fit,variable_old)
    checks["minimum_pairwise_output_distance_for_distinct_old_rows"]=float(
        min(np.linalg.norm(variable_pred[i]-variable_pred[j]) for i in range(4) for j in range(i)))
    assert checks["minimum_pairwise_output_distance_for_distinct_old_rows"] > .99

    checks["rotation_shape"]=list(fit.rotation.shape)
    checks["output_shape"]=list(pred.shape)
    assert fit.rotation.shape==(384,768) and pred.shape[1]==768

    failed=0
    for fn in [
        lambda: fit_rectangular_procrustes(np.zeros((3,5)),np.zeros((3,4))),
        lambda: fit_rectangular_procrustes(np.zeros((3,4)),np.zeros((4,5))),
        lambda: fit_rectangular_procrustes(np.zeros((3,4)),np.zeros((3,5)),old_pilot=np.zeros((2,4))),
    ]:
        try:
            fn()
        except ValueError:
            failed += 1
    checks["invalid_cases_rejected"]=failed
    assert failed==3

    out={"study":"procrustes_alignment_01",
         "scope":"Synthetic mathematical/interface checks only; no DINO features, R1/R2, or paired025 execution.",
         "checks":checks,"passed":True,
         "not_claimed":["stronger DINO frontend","FastFill reproduction","revision improvement","resource advantage","method novelty"]}
    Path("RESULTS.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
