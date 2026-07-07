#!/usr/bin/env python3
"""Smoke tests for the release pipeline.

Run from the repository root:

    python -m pytest tests/        # if pytest is installed
    python tests/test_smoke.py     # standalone

CPU-only checks: (1) the train-derived template is well-formed; (2) no API key
or secret is committed to the repository (the LLM-assist module reads its key
from the environment only).
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'src'
sys.path.insert(0, str(SRC))


class TestTemplateAntiSymmetry(unittest.TestCase):
    """The 54x54 train-derived template must satisfy T[i, j] = -T[j, i]
    (heading and range): entry (i, j) is the relative pose from orbital
    position i to j."""

    def setUp(self):
        from per_pair_finalize import load_template
        template_path = HERE.parent / 'assets' / 'template.json'
        self.assertTrue(template_path.exists(),
                        f'template.json missing at {template_path}')
        self.T = load_template(template_path)

    def test_shape(self):
        self.assertEqual(self.T.shape, (54, 54, 2))

    def test_diagonal_zero(self):
        self.assertLess(np.abs(np.diag(self.T[:, :, 0])).max(), 1e-6)
        self.assertLess(np.abs(np.diag(self.T[:, :, 1])).max(), 1e-6)

    def test_heading_antisymmetry(self):
        def cd(a, b):
            return ((a - b + 180) % 360) - 180
        err = cd(self.T[:, :, 0], -self.T[:, :, 0].T)
        self.assertLess(np.abs(err).max(), 1e-6)

    def test_range_antisymmetry(self):
        err = self.T[:, :, 1] + self.T[:, :, 1].T
        self.assertLess(np.abs(err).max(), 1e-6)


class TestNoSecret(unittest.TestCase):
    """Security check: no API key may be committed. The LLM-assist module must
    obtain its key from the environment. Scans by pattern (this file contains
    no real key itself)."""

    import re as _re
    SECRET_PATTERNS = (
        _re.compile(r'sk-[A-Za-z0-9_\-]{20,}'),
        _re.compile(r'(?i)(api[_-]?key|secret|token)\s*[=:]\s*["\'][A-Za-z0-9_\-]{16,}["\']'),
        _re.compile(r'Bearer\s+[A-Za-z0-9_\-\.]{20,}'),
    )

    def test_no_hardcoded_secret(self):
        pkg_root = HERE.parent
        bad = []
        for path in pkg_root.rglob('*'):
            if path.is_dir() or path.suffix == '.pyc' or '.git' in path.parts:
                continue
            if path.name == os.path.basename(__file__):
                continue
            try:
                text = path.read_text(errors='ignore')
            except (UnicodeDecodeError, PermissionError):
                continue
            for pat in self.SECRET_PATTERNS:
                if pat.search(text):
                    bad.append(f'{path.relative_to(pkg_root)}: {pat.pattern[:24]}')
        self.assertEqual([], bad,
                         'possible hard-coded secret in repository:\n  ' + '\n  '.join(bad))

    def test_llm_assist_reads_key_from_env(self):
        src = (HERE.parent / 'src' / 'llm_assist.py').read_text()
        self.assertIn("os.environ.get('OPENAI_API_KEY')", src)
        self.assertNotIn('sk-', src.replace('sk-...', ''))


if __name__ == '__main__':
    unittest.main(verbosity=2)
