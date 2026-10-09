"""Run source-parity and integration checks. No data download or real-model training.

Usage:
  python run_checks.py --upstream-dir /path/to/read-back/ml-fct --out RESULTS.json
Expected upstream files can be nested as in ml-fct or flat. Their exact blob SHAs
are checked before AST-extracting the one reference method to avoid executing
training entrypoints or the rest of the external project.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
import numpy as np
import torch
from frontend_interface import (
    fastfill_loss, split_output, validate_preparation_boundary, plan_backfill,
    assert_same_evidence_receipts,
)

EXPECTED = {
    'utils/objectives.py': 'c8c6a3efbc1c83ec057b0b8349dfdc9b52edaf09',
    'trainers/transformation_trainer.py': '649a21af754de646abc084f79771d4510efd03f4',
}


def load_reference(root: Path):
    trees, hashes = {}, {}
    for rel, sha in EXPECTED.items():
        p = root / rel
        if not p.is_file():
            p = root / Path(rel).name
        if not p.is_file():
            raise FileNotFoundError(f'Missing pinned upstream source: {rel}')
        b = p.read_bytes()
        observed = hashlib.sha1(f'blob {len(b)}\0'.encode() + b).hexdigest()
        if observed != sha:
            raise ValueError(f'Upstream version mismatch: {rel}: {observed}')
        hashes[rel] = {'git_blob_sha': observed, 'bytes': len(b)}
        trees[rel] = ast.parse(b.decode('utf-8'))
    ns = {'torch': torch, 'nn': torch.nn}
    classes = [n for n in trees['utils/objectives.py'].body
               if isinstance(n, ast.ClassDef) and n.name in ('UncertaintyLoss', 'LabelSmoothing')]
    exec(compile(ast.Module(body=classes, type_ignores=[]), '<pinned-objectives>', 'exec'), ns)
    cls = next(n for n in trees['trainers/transformation_trainer.py'].body
               if isinstance(n, ast.ClassDef) and n.name == 'TransformationTrainer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_loss')
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<pinned-compute-loss>', 'exec'), ns)
    return ns, hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('RESULTS.json'))
    args = parser.parse_args()
    reference, hashes = load_reference(args.upstream_dir)
    torch.set_num_threads(2)
    parity = []
    diagnostics = {}

    class Checks(unittest.TestCase):
        def test_01_full_loss_and_gradients_match_read_back_source(self):
            for n, d in [(2, 4), (9, 128), (8, 768)]:
                with self.subTest(batch=n, target_dim=d):
                    gen = torch.Generator().manual_seed(611000 + n + d)
                    target = torch.randn(n, d, generator=gen, dtype=torch.float64)
                    labels = torch.arange(n) % 5
                    head = torch.nn.Linear(d, 5, dtype=torch.float64)
                    with torch.no_grad():
                        head.weight.copy_(torch.randn(5, d, generator=gen, dtype=torch.float64) / d ** 0.5)
                        head.bias.copy_(torch.randn(5, generator=gen, dtype=torch.float64))
                    head.requires_grad_(False)
                    head.eval()
                    raw_a = torch.randn(n, d + 1, generator=gen, dtype=torch.float64).requires_grad_(True)
                    raw_b = raw_a.detach().clone().requires_grad_(True)
                    holder = SimpleNamespace(
                        new_model=SimpleNamespace(model=SimpleNamespace(fc=head)),
                        criterion_similarity=torch.nn.MSELoss(reduction='none'),
                        criterion_disc=reference['LabelSmoothing'](smoothing=0.1, reduction='none'),
                        criterion_uncertainty=reference['UncertaintyLoss'](mu_uncertainty=0.5),
                        mu_similarity=1.0, mu_disc=1.0,
                    )
                    source_loss = reference['compute_loss'](holder, target, raw_a[:, :d], labels, raw_a[:, d:])
                    new_loss = fastfill_loss(raw_b[:, :d], target, labels, raw_b[:, d:], head)
                    source_grad = torch.autograd.grad(source_loss, raw_a)[0]
                    new_grad = torch.autograd.grad(new_loss, raw_b)[0]
                    le = abs(float(source_loss.detach() - new_loss.detach()))
                    ge = float((source_grad - new_grad).abs().max())
                    parity.append({'batch': n, 'target_dim': d, 'loss_abs_error': le, 'gradient_max_abs_error': ge})
                    self.assertLessEqual(le, 1e-12)
                    self.assertLessEqual(ge, 1e-12)

        def test_02_target_feature_and_uncertainty_coordinates_are_separate(self):
            raw = np.arange(3 * 769, dtype=np.float32).reshape(3, 769)
            z, s = split_output(raw, 768)
            self.assertEqual(z.shape, (3, 768))
            self.assertEqual(s.shape, (3,))
            np.testing.assert_array_equal(z, raw[:, :768])
            self.assertFalse(z.flags.writeable)
            raw[:] = -1
            self.assertEqual(z[0, 1], 1)

        def test_03_no_silent_dimension_truncation(self):
            with self.assertRaises(ValueError):
                split_output(np.zeros((3, 769), dtype=np.float32), 384)

        def test_04_nonfinite_output_is_rejected(self):
            with self.assertRaises(ValueError):
                split_output(np.array([[0., np.inf]]), 1)

        def test_05_uncertainty_order_known_exclusion_and_ties(self):
            p = plan_backfill([10, 20, 30, 40], [2., 5., 2., -1.], [20], [1, 2])
            self.assertEqual(p.stages, ((10,), (30, 40)))
            self.assertEqual(p.unspent_allowances, (0, 0))

        def test_06_extreme_log_uncertainty_does_not_require_exponentiation(self):
            p = plan_backfill([1, 2, 3], [1000., -1000., 0.], [], [3])
            self.assertEqual(p.acquired_ids, (1, 3, 2))

        def test_07_common_order_is_not_silently_replaced_by_native_order(self):
            p = plan_backfill([1, 2, 3], [8., 2., 5.], [], [2], common_order=[2, 1, 3])
            self.assertEqual(p.acquired_ids, (2, 1))
            self.assertEqual(p.policy, 'common_frozen_order')

        def test_08_allowances_are_unique_and_unused_allowance_is_visible(self):
            p = plan_backfill([1, 2], [1., 2.], [1], [0, 1, 3])
            self.assertEqual(p.stages, ((), (2,), ()))
            self.assertEqual(p.unspent_allowances, (0, 0, 3))

        def test_09_same_count_but_different_evidence_is_rejected(self):
            a = [('upgrade', 1, 'v2', 'labels-v0', 'payload-a')]
            b = [('upgrade', 2, 'v2', 'labels-v0', 'payload-b')]
            with self.assertRaises(ValueError):
                assert_same_evidence_receipts(a, b)
            assert_same_evidence_receipts(a, list(a))

        def test_10_supervised_preparation_split_is_checked(self):
            validate_preparation_boundary([1, 2], [3, 4], [5, 6])
            with self.assertRaises(ValueError):
                validate_preparation_boundary([1, 2], [2, 3], [5, 6])

        def test_11_invalid_plans_are_rejected(self):
            for ids, s, known, budgets, order in [
                ([1, 1], [0., 0.], [], [1], None),
                ([1, 2], [0., 0.], [], [-1], None),
                ([1, 2], [0., 0.], [3], [1], None),
                ([1, 2], [0., 0.], [], [1], [1]),
                ([1, 2], [np.nan, 0.], [], [1], None),
            ]:
                with self.subTest(ids=ids, budgets=budgets, order=order):
                    with self.assertRaises(ValueError):
                        plan_backfill(ids, s, known, budgets, common_order=order)

        def test_12_log_uncertainty_optimum_tracks_combined_loss(self):
            # Standard calculus for fixed positive base loss, not a trained-model result.
            base_losses = torch.tensor([1., 4.], dtype=torch.float64)
            mu = 0.5
            optimum = torch.log(base_losses / (2 * mu)).requires_grad_(True)
            objective = (0.5 * torch.exp(-optimum) * base_losses + mu * optimum).sum()
            grad = torch.autograd.grad(objective, optimum)[0]
            self.assertLess(float(grad.abs().max()), 1e-12)
            diagnostics['conditional_scalar_optimum'] = {
                'base_combined_losses': base_losses.tolist(),
                'exp_s_at_optimum': torch.exp(optimum.detach()).tolist(),
                'gradient_max_abs': float(grad.abs().max()),
                'meaning': 'Fixed positive combined loss optimum; not feature-covariance calibration.'}

    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        'study': 'fastfill_interface_01',
        'scope': 'Synthetic component integration and public-source parity only; no real-model results.',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'torch': torch.__version__, 'device': 'cpu', 'threads': 2},
        'source_files': hashes,
        'tests_run': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors), 'success': result.wasSuccessful(),
        'parity': parity, 'diagnostics': diagnostics,
        'elapsed_seconds_not_deployment_cost': time.perf_counter() - start,
        'not_performed': ['DINO inference', 'MLP/backbone training', 'new R1/R2 evaluation',
                          'old paired025 core execution', 'native FastFill retrieval reproduction',
                          'real-data source/output recovery', 'full resource comparison'],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    raise SystemExit(main())
