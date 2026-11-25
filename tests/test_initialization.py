import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from historical_swedish.train import initialize, parser


class InitializationTests(unittest.TestCase):
    def test_huggingface_and_local_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'weights.pth'
            source = torch.nn.Linear(3, 4)
            target = torch.nn.Linear(3, 4)
            torch.save({'model': source.state_dict()}, path)
            with patch('huggingface_hub.hf_hub_download', return_value=str(path)) as download:
                initialize(target, 'blowing-up-groundhogs/eruku', revision='pinned-commit')
            download.assert_called_once_with(repo_id='blowing-up-groundhogs/eruku',
                                            filename='000073688.pth', revision='pinned-commit')
            self.assertTrue(torch.equal(source.weight, target.weight))
            initialize(target, path)
            with self.assertRaises(FileNotFoundError):
                initialize(target, Path(directory)/'missing.pth')

    def test_parser_preserves_repo_id(self):
        args = parser().parse_args(['--manifest', 'data.jsonl', '--output-dir', 'output',
                                    '--init-checkpoint', 'blowing-up-groundhogs/eruku'])
        self.assertEqual(args.init_checkpoint, 'blowing-up-groundhogs/eruku')
