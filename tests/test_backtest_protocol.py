"""Fail-closed provenance gate tests without competition data or Parquet."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from merry_llama import backtest_protocol as b
from merry_llama.data import training_names


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.paths = [self.base/n for n in training_names()]
        for p in self.paths: p.write_bytes(p.name.encode())
        self.audit = {'contract': b.CONTRACT, 'reviewer': 'synthetic',
            'model_selection': 'published_frozen_exploratory',
            'base_sha256': b.inventory(self.paths, self.base),
            'checks': {k: {'approved': True, 'evidence': 'synthetic only'} for k in b.CHECKS}}
        self.path = self.base/'audit.json'

    def validate(self):
        self.path.write_text(json.dumps(self.audit))
        return b.validate_audit(self.base, self.path)

    def test_complete_attestation(self):
        self.assertEqual(self.validate(), self.audit)

    def test_every_check_required(self):
        for k in b.CHECKS:
            for invalid in [False, 'true', None]:
                with self.subTest(check=k, value=invalid):
                    self.audit['checks'][k]['approved'] = invalid
                    with self.assertRaisesRegex(ValueError, k): self.validate()
            self.audit['checks'][k]['approved'] = True
            self.audit['checks'][k]['evidence'] = ' '
            with self.assertRaisesRegex(ValueError, k): self.validate()
            self.audit['checks'][k]['evidence'] = 'synthetic only'

    def test_content_mutation_rejected(self):
        self.paths[0].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256'): self.validate()

    def test_extra_inventory_and_missing_input_rejected(self):
        self.audit['base_sha256']['ranking.parquet'] = 'forbidden'
        with self.assertRaisesRegex(ValueError, 'SHA-256'): self.validate()
        self.paths[0].unlink()
        with self.assertRaises(FileNotFoundError): self.validate()

    def test_selection_cannot_claim_independence(self):
        self.audit['model_selection'] = 'independent'
        with self.assertRaisesRegex(ValueError, 'published_frozen'): self.validate()


if __name__ == '__main__': unittest.main()
