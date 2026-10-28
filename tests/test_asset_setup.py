import json
from pathlib import Path
import tempfile
import unittest

from g1_deploy.catalog import ROOT, catalog
from scripts.download_models import link_models


class AssetSetupTests(unittest.TestCase):
    def test_model_links_are_idempotent_and_preserve_existing_data(self):
        spec = {'source_path': 'models/source', 'model_path': 'models/by-family/test'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / spec['source_path']
            source.mkdir(parents=True)
            (source / 'weights').write_bytes(b'checkpoint')
            destination = root / spec['model_path']
            link_models({'test': spec}, root)
            link_models({'test': spec}, root)
            self.assertEqual((destination / 'weights').read_bytes(), b'checkpoint')
            destination.unlink()
            destination.symlink_to(root / 'elsewhere')
            with self.assertRaisesRegex(RuntimeError, 'points elsewhere'):
                link_models({'test': spec}, root)
            destination.unlink()
            destination.mkdir()
            (destination / 'keep').write_text('existing data')
            with self.assertRaisesRegex(RuntimeError, 'overwrite'):
                link_models({'test': spec}, root)
            self.assertEqual((destination / 'keep').read_text(), 'existing data')

    def test_galaxea_inventory_covers_all_catalog_assets(self):
        inventory = json.loads((ROOT / 'dependencies/galaxea-source-inventory.json').read_text())
        files = {f['rfilename']: f for f in inventory['files']}
        for spec in catalog()['models'].values():
            if spec['family'] != 'galaxea':
                continue
            # Catalog entries may pin older revisions with identical assets.
            # Check their content digests, not equality of repository revisions.
            for expected in spec['weight_files']:
                item = files[spec['folder'] + '/' + expected['path']]
                self.assertEqual(item['lfs']['sha256'], expected['sha256'])
            for expected in spec['required_files']:
                item = files[spec['folder'] + '/' + expected['path']]
                self.assertEqual(item['size'], expected['size'])
                self.assertTrue(item.get('lfs', {}).get('sha256') or item.get('blobId'))
        processor = json.loads((ROOT / 'dependencies/galaxea-processor-inventory.json').read_text())
        self.assertTrue(processor['revision'])
        self.assertTrue(processor['files'])
        for item in processor['files']:
            self.assertTrue(item['rfilename'].startswith('qwen3_5_2b_base_processor/'))
            self.assertTrue(item.get('lfs', {}).get('sha256') or item.get('blobId'))
