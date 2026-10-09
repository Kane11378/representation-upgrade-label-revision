"""Small FastFill integration boundary; not a trained model or a new algorithm.

This module never changes the established paired025 implementation.  It separates:
  * the public scalar-uncertainty loss;
  * target-dimensional features from the extra log-variance output;
  * a precomputed backfill order from the downstream estimator.
The caller remains responsible for provenance, acquisition and all resource costs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np
import torch
from torch import nn


def fastfill_loss(
    transformed: torch.Tensor,
    target_features: torch.Tensor,
    labels: torch.Tensor,
    log_variance: torch.Tensor,
    frozen_head: nn.Module,
    *,
    smoothing: float = 0.1,
    mu_similarity: float = 1.0,
    mu_disc: float = 1.0,
    mu_uncertainty: float = 0.5,
) -> torch.Tensor:
    """Match the read-back public CIFAR FastFill loss on nonsingleton batches.

    MSE is averaged over feature coordinates *per example*, then combined with
    label-smoothed CE before uncertainty weighting. No feature normalization is
    silently added. The head must already be frozen, identified and costed.
    Its supervised preparation labels are NOT provided by this function.
    """
    if transformed.ndim != 2 or transformed.shape != target_features.shape:
        raise ValueError('Expected equally shaped (N,D) feature tensors.')
    n, d = transformed.shape
    if n < 2 or d < 2:
        raise ValueError('This parity-checked boundary requires N>=2 and D>=2.')
    if labels.shape != (n,) or labels.dtype != torch.long:
        raise ValueError('labels must be a torch.long vector with N elements.')
    if log_variance.shape not in ((n,), (n, 1)):
        raise ValueError('Exactly one log-variance output per example is required.')
    if any(p.requires_grad for p in frozen_head.parameters()):
        raise ValueError('Freeze the supplied classification head first.')
    if any(m.training for m in frozen_head.modules()):
        raise ValueError('Put the frozen classification head in evaluation mode first.')
    if not 0 <= smoothing <= 1:
        raise ValueError('smoothing must belong to [0,1].')
    if (not all(math.isfinite(v) for v in
                (mu_similarity, mu_disc, mu_uncertainty))
        or min(mu_similarity, mu_disc) < 0 or mu_uncertainty <= 0):
        raise ValueError('Invalid loss coefficients.')
    target_features = target_features.detach()
    per_example_mse = (transformed - target_features).square().mean(dim=1)
    logits = frozen_head(transformed)
    if logits.ndim != 2 or logits.shape[0] != n or logits.shape[1] < 2:
        raise ValueError('The head must return (N,C), C>=2.')
    logp = torch.log_softmax(logits, dim=-1)
    nll = -logp.gather(1, labels[:, None])[:, 0]
    smoothed_ce = (1.0 - smoothing) * nll - smoothing * logp.mean(dim=1)
    loss_vector = mu_similarity * per_example_mse + mu_disc * smoothed_ce
    s = log_variance.reshape(n)
    return (0.5 * torch.exp(-s) * loss_vector + mu_uncertainty * s).mean()


def split_output(raw: np.ndarray, target_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Detach the extra uncertainty coordinate; never truncate target features."""
    a = np.asarray(raw)
    if not isinstance(target_dim, int) or target_dim < 1:
        raise ValueError('target_dim must be positive.')
    if a.ndim != 2 or a.shape[1] != target_dim + 1 or a.shape[0] == 0:
        raise ValueError('Expected exactly N x (target_dim + 1), N>0.')
    if not np.issubdtype(a.dtype, np.floating) or not np.isfinite(a).all():
        raise ValueError('Expected finite floating point outputs.')
    features, logvar = a[:, :target_dim].copy(), a[:, target_dim].copy()
    features.flags.writeable = False
    logvar.flags.writeable = False
    return features, logvar


def _ids(values: Sequence[int] | np.ndarray, name: str) -> tuple[int, ...]:
    vals = tuple(values)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           for v in vals):
        raise ValueError(f'{name}: integer IDs in one declared dataset namespace required.')
    out = tuple(int(v) for v in vals)
    if len(out) != len(set(out)):
        raise ValueError(f'{name}: duplicate IDs.')
    return out


def validate_preparation_boundary(
    supervised_preparation_ids: Sequence[int],
    revisable_ids: Sequence[int],
    downstream_eval_ids: Sequence[int],
) -> None:
    """A proposed downstream-study split, NOT FastFill's native gallery/query split.

    Preparation includes both head-training and alignment-training IDs. This is
    only a check of the supplied controlled-study manifest. It is not proof that
    an external encoder has never encountered these images during pretraining.
    """
    groups = [set(_ids(v, name)) for v, name in (
        (supervised_preparation_ids, 'preparation'),
        (revisable_ids, 'history'), (downstream_eval_ids, 'evaluation'))]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('The proposed independent downstream split overlaps.')


@dataclass(frozen=True)
class AcquisitionPlan:
    policy: str
    initial_exact: tuple[int, ...]
    stages: tuple[tuple[int, ...], ...]
    unspent_allowances: tuple[int, ...]

    @property
    def acquired_ids(self) -> tuple[int, ...]:
        return tuple(i for stage in self.stages for i in stage)


def plan_backfill(
    record_ids: Sequence[int],
    log_variance: Sequence[float],
    initial_exact_ids: Sequence[int],
    stage_allowances: Sequence[int],
    *,
    common_order: Sequence[int] | None = None,
) -> AcquisitionPlan:
    """Freeze discretionary backfill IDs for reuse by BOTH mean and paired025.

    No true target features, corrected labels, scores or future requests enter
    this API. Availability is encoded by the caller's eligible record_ids.
    This handles only proactive/discretionary backfill, not request-time retrieval.
    Request-time access must be shared and accounted separately in the old engine.
    Equal-score tie-breaking by ID is an explicit deterministic adapter convention.
    """
    ids = _ids(record_ids, 'record_ids')
    known = _ids(initial_exact_ids, 'initial_exact_ids')
    s = np.asarray(log_variance, dtype=np.float64)
    if not ids or s.shape != (len(ids),) or not np.isfinite(s).all():
        raise ValueError('One finite log-variance is required per eligible record.')
    if not set(known) <= set(ids):
        raise ValueError('Initial exact IDs must be within the eligible manifest.')
    allowances = tuple(stage_allowances)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
           or x < 0 for x in allowances):
        raise ValueError('Stage allowances must be nonnegative integers.')
    if common_order is None:
        # Sorting log-variance gives the same order as exp(log-variance) without overflow.
        order = tuple(ids[j] for j in sorted(range(len(ids)), key=lambda j: (-s[j], ids[j])))
        policy = 'fastfill_predicted_uncertainty'
    else:
        order = _ids(common_order, 'common_order')
        if set(order) != set(ids):
            raise ValueError('common_order must be a complete permutation of eligible IDs.')
        policy = 'common_frozen_order'
    known_set = set(known)
    remaining = [i for i in order if i not in known_set]
    cursor = 0
    stages, unspent = [], []
    for allowance in allowances:
        take = tuple(remaining[cursor:cursor + int(allowance)])
        cursor += len(take)
        stages.append(take)
        unspent.append(int(allowance) - len(take))
    return AcquisitionPlan(policy, tuple(sorted(known)), tuple(stages), tuple(unspent))


def assert_same_evidence_receipts(
    mean_receipts: Sequence[tuple], paired_receipts: Sequence[tuple]
) -> None:
    """Compare recorded actual access, not only the numerical quota.

    Each caller-supplied receipt should identify stage, ID, feature/label version,
    and the payload digest. Equality of logs is not an independent hardware audit.
    """
    if tuple(mean_receipts) != tuple(paired_receipts):
        raise ValueError('The two backends did not record the same evidence sequence.')
