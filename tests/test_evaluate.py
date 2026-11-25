import unittest
from unittest.mock import patch
from historical_swedish.evaluate import edit_distance, error_rates, exposure


class EvaluationTests(unittest.TestCase):
    def test_character_and_word_errors_use_corpus_denominators(self):
        rows = [dict(text='åäö', pred='åö'), dict(text='en lång rad', pred='en lång fel')]
        result = error_rates(rows, 'pred')
        self.assertAlmostEqual(result['cer'], 4/14)
        self.assertAlmostEqual(result['wer'], 2/4)

    def test_unicode_normalization_and_empty_generation(self):
        self.assertEqual(error_rates([dict(text='å', pred='a\u030a')], 'pred')['cer'], 0)
        self.assertEqual(error_rates([dict(text='borta', pred='')], 'pred')['cer'], 1)
        self.assertEqual(edit_distance('abc', 'abcabc'), 3)

    def test_author_fid_adapter_matches_analytic_gaussian_distance(self):
        try:
            from historical_swedish.evaluate import configure_hwd_scipy
            configure_hwd_scipy()
            from hwd.metrics.fid.fid_score_crop64x64 import calculate_frechet_distance
            import numpy as np
        except ImportError:
            self.skipTest('optional HWD dependencies are not installed')
        # Equal unit covariances cancel, leaving squared mean distance 1² + 2².
        self.assertAlmostEqual(calculate_frechet_distance(
            np.array([0., 0.]), np.eye(2), np.array([1., 2.]), np.eye(2)), 5.)

    def test_pass_count_accounts_for_partial_accumulation(self):
        progress = dict(stage=0, step=5, height=64, data_epoch=1,
            stages=[dict(name='long', steps=10, style_words=[4,8], target_words=[1,32], max_width=2048)],
            args=dict(manifest='unused', curriculum='eruku', pair_by='page_id', use='normalized',
                      historical_probability=.8, replay_train_pattern='*.tar', batch_size=2, accumulation=3))
        with patch('historical_swedish.data.LinePairs', return_value=list(range(16))):
            result = exposure(progress, 1)
        # 20 mixed pairs = 10 microbatches = 4 optimizer updates/pass.
        self.assertEqual(result['optimizer_updates_per_pass'], 4)
        self.assertAlmostEqual(result['estimated_passes'], 1.3)


if __name__ == '__main__':
    unittest.main()
