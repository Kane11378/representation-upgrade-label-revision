"""Two current-evidence estimators, not an initial-label predictor plus a patch.

This is a contract-repair prototype built from standard ridge/model averaging.
It is exact for its declared state, not for unobserved target features. Old
labels may exist in external audit logs, but this state does not read them.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def cross(z: np.ndarray, y: np.ndarray, classes: int) -> np.ndarray:
    ans = np.zeros((z.shape[1], classes), dtype=np.float64)
    np.add.at(ans.T, y, z)
    return ans


def solve(q: np.ndarray, h: np.ndarray) -> np.ndarray:
    return cho_solve(cho_factor((q + q.T) / 2, lower=True, check_finite=True), h)


def integer_vector(x, name):
    a = np.asarray(x)
    if a.ndim != 1 or a.dtype.kind not in 'iu':
        raise ValueError(f'{name} must be a one-dimensional integer array')
    return a.astype(np.int64, copy=True)


@dataclass(frozen=True)
class LabelChange:
    row: int
    expected_version: int
    old_label: int
    new_label: int


class CurrentEvidenceState:
    """Copy-owned inputs. Pure label updates reuse two current factorizations.

    Certainty is evidence provenance, not 'was historically mislabelled'. A
    verified unchanged label receives the same role as a changed label.
    Fixed pilot rows remaining outside certainty represent the remaining
    finite population. No sampling/unlearning guarantee is claimed here.
    """
    def __init__(self, proxy, labels, known, observed, certainty, pilot,
                 classes, ridge=1., shrink=.1, feature_version='v2'):
        self.proxy = np.asarray(proxy, dtype=np.float64).copy()
        self.labels = integer_vector(labels, 'labels')
        self.known = np.asarray(known, dtype=bool).copy()
        self.observed = np.asarray(observed, dtype=np.float64).copy()
        self.certainty = np.asarray(certainty, dtype=bool).copy()
        self.pilot = integer_vector(pilot, 'pilot')
        self.classes, self.ridge, self.shrink = int(classes), float(ridge), float(shrink)
        self.feature_version = str(feature_version)
        self.versions = np.zeros(len(self.labels), dtype=np.int64)
        self._validate()
        self._build()

    def _validate(self):
        if self.proxy.ndim != 2 or self.proxy.shape[1] < 2:
            raise ValueError('features must be a matrix including final bias coordinate')
        n, d = self.proxy.shape
        if (self.observed.shape != (n, d) or self.labels.shape != (n,)
                or self.known.shape != (n,) or self.certainty.shape != (n,)):
            raise ValueError('shape mismatch')
        if not np.isfinite(self.proxy).all() or not np.isfinite(self.observed).all():
            raise ValueError('nonfinite feature')
        if self.classes < 2 or self.ridge <= 0 or not np.isfinite(self.ridge):
            raise ValueError('invalid classes/ridge')
        if not 0 <= self.shrink <= 1 or not self.feature_version:
            raise ValueError('invalid shrink/version')
        if np.any(self.labels < 0) or np.any(self.labels >= self.classes):
            raise ValueError('invalid current label')
        if np.any(self.observed[~self.known] != 0):
            raise ValueError('unqueried target values supplied')
        if np.any(self.certainty & ~self.known):
            raise ValueError('certainty requires an observed feature')
        if (len(np.unique(self.pilot)) != len(self.pilot)
                or np.any(self.pilot < 0) or np.any(self.pilot >= n)
                or np.any(~self.known[self.pilot])):
            raise ValueError('invalid or unknown pilot')
        if not np.all(self.proxy[:, -1] == 1) or not np.all(self.observed[self.known, -1] == 1):
            raise ValueError('final feature coordinate must be constant one')
        if not self.known.all() and not np.any(~self.certainty[self.pilot]):
            raise ValueError('NEEDS_MORE_EVIDENCE: no pilot in remaining population')

    def _build(self):
        n, d = self.proxy.shape
        self.means = self.proxy.copy()
        self.means[self.known] = self.observed[self.known]
        self.q_mean = self.means.T @ self.means + self.ridge * np.eye(d)
        self.h_mean = cross(self.means, self.labels, self.classes)
        self.sample_weights = np.zeros(n)
        if self.known.all():
            # Full observation must not keep artificial sampling shrinkage.
            self.sample_weights[:] = 1
            self.q_sample = self.q_mean.copy()
        else:
            remaining = ~self.certainty
            p = self.pilot[remaining[self.pilot]]
            nr = int(remaining.sum())
            mu = self.observed[p, :-1].mean(0)
            residual = self.observed[p, :-1] - mu
            cov = residual.T @ residual / len(p)
            cov = ((1 - self.shrink) * cov
                   + self.shrink * np.trace(cov) / (d - 1) * np.eye(d - 1))
            second = np.zeros((d, d))
            second[:-1, :-1] = cov + np.outer(mu, mu)
            second[:-1, -1] = mu
            second[-1, :-1] = mu
            second[-1, -1] = 1
            self.q_sample = (self.observed[self.certainty].T @ self.observed[self.certainty]
                             + nr * second + self.ridge * np.eye(d))
            self.sample_weights[self.certainty] = 1
            self.sample_weights[p] = nr / len(p)
        self.h_sample = cross(self.observed * self.sample_weights[:, None],
                              self.labels, self.classes)
        self._cm = cho_factor(self.q_mean, lower=True)
        self._cs = cho_factor(self.q_sample, lower=True)
        self.w_mean = cho_solve(self._cm, self.h_mean)
        self.w_sample = cho_solve(self._cs, self.h_sample)

    def weights(self, alpha=.25, *, paired=True):
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('alpha must lie in [0,1]')
        second = self.w_sample if paired else cho_solve(self._cs, self.h_mean)
        return (1 - alpha) * self.w_mean + alpha * second

    def relabel(self, changes, *, feature_version):
        """Validate whole request before updating. Feature evidence stays fixed."""
        if feature_version != self.feature_version:
            raise ValueError('feature version mismatch')
        changes = list(changes)
        if len({r.row for r in changes}) != len(changes):
            raise ValueError('duplicate row in request')
        for r in changes:
            values = (r.row, r.expected_version, r.old_label, r.new_label)
            if any(not isinstance(x, (int, np.integer)) or isinstance(x, bool) for x in values):
                raise ValueError('integer change fields required')
            if not 0 <= r.row < len(self.labels):
                raise ValueError('unknown row')
            if self.versions[r.row] != r.expected_version or self.labels[r.row] != r.old_label:
                raise ValueError('stale current label/version')
            if not 0 <= r.new_label < self.classes:
                raise ValueError('invalid new label')
        dm = np.zeros_like(self.h_mean)
        ds = np.zeros_like(self.h_sample)
        for r in changes:
            dm[:, r.old_label] -= self.means[r.row]
            dm[:, r.new_label] += self.means[r.row]
            z = self.sample_weights[r.row] * self.observed[r.row]
            ds[:, r.old_label] -= z
            ds[:, r.new_label] += z
        hm, hs = self.h_mean + dm, self.h_sample + ds
        wm, ws = cho_solve(self._cm, hm), cho_solve(self._cs, hs)
        if not np.isfinite(wm).all() or not np.isfinite(ws).all():
            raise FloatingPointError('solve failed; state unchanged')
        self.h_mean, self.h_sample, self.w_mean, self.w_sample = hm, hs, wm, ws
        for r in changes:
            self.labels[r.row] = r.new_label
            self.versions[r.row] += 1

    def observe(self, rows, values, *, feature_version, authoritative=True):
        """Refresh evidence and rebuild both consistent branches, before relabeling.

        A no-op label verification can call this identically. Existing exact
        values cannot be silently changed inside one feature version.
        """
        if feature_version != self.feature_version:
            raise ValueError('feature version mismatch')
        rows = integer_vector(rows, 'rows')
        vals = np.asarray(values, dtype=np.float64)
        if (len(np.unique(rows)) != len(rows) or np.any(rows < 0)
                or np.any(rows >= len(self.labels))
                or vals.shape != (len(rows), self.proxy.shape[1])
                or not np.isfinite(vals).all() or not np.all(vals[:, -1] == 1)):
            raise ValueError('invalid evidence request')
        previous = self.known[rows]
        if np.any(previous) and not np.array_equal(vals[previous], self.observed[rows[previous]]):
            raise ValueError('conflicting exact values require an explicit new feature version')
        known, obs, certainty = self.known.copy(), self.observed.copy(), self.certainty.copy()
        known[rows] = True
        obs[rows] = vals
        if authoritative:
            certainty[rows] = True
        new = CurrentEvidenceState(self.proxy, self.labels, known, obs, certainty,
                                   self.pilot, self.classes, self.ridge, self.shrink,
                                   self.feature_version)
        new.versions = self.versions.copy()
        self.__dict__.update(new.__dict__)

    def append(self, values, labels, *, feature_version):
        if feature_version != self.feature_version:
            raise ValueError('feature version mismatch')
        values = np.asarray(values, dtype=np.float64)
        labels = integer_vector(labels, 'labels')
        if values.ndim != 2 or values.shape != (len(labels), self.proxy.shape[1]):
            raise ValueError('invalid append shape')
        new = CurrentEvidenceState(np.vstack((self.proxy, values)), np.r_[self.labels, labels],
                                   np.r_[self.known, np.ones(len(labels), bool)],
                                   np.vstack((self.observed, values)),
                                   np.r_[self.certainty, np.ones(len(labels), bool)],
                                   self.pilot, self.classes, self.ridge, self.shrink,
                                   self.feature_version)
        new.versions = np.r_[self.versions, np.zeros(len(labels), np.int64)]
        self.__dict__.update(new.__dict__)

    def numeric_payload(self):
        # Array payload of this straightforward implementation, not process RSS.
        return {k: int(v.nbytes) for k, v in self.__dict__.items() if isinstance(v, np.ndarray)} | {
            'mean_cholesky': int(self._cm[0].nbytes), 'sample_cholesky': int(self._cs[0].nbytes)}
