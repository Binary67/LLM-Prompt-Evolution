import unittest
from unittest.mock import patch
import pandas as pd
from PromptEvaluation import EvaluatePrompt

class MetricsTestCase(unittest.TestCase):
    def test_perfect_predictions(self):
        DataFrame = pd.DataFrame({'text': ['a', 'b'], 'label': ['yes', 'no']})
        Predictions = iter(['yes', 'no'])
        with patch('PromptEvaluation.GenerateOutput', side_effect=lambda Prompt, **Variables: next(Predictions)):
            Metrics, _ = EvaluatePrompt('template', DataFrame, ['text'], 'label')
        self.assertAlmostEqual(Metrics['Accuracy'], 1.0)
        self.assertAlmostEqual(Metrics['Precision'], 1.0)
        self.assertAlmostEqual(Metrics['Recall'], 1.0)
        self.assertAlmostEqual(Metrics['F1'], 1.0)

    def test_partial_predictions(self):
        DataFrame = pd.DataFrame({'text': ['a', 'b', 'c'], 'label': ['yes', 'no', 'yes']})
        Predictions = iter(['no', 'no', 'yes'])
        with patch('PromptEvaluation.GenerateOutput', side_effect=lambda Prompt, **Variables: next(Predictions)):
            Metrics, _ = EvaluatePrompt('template', DataFrame, ['text'], 'label')
        self.assertAlmostEqual(Metrics['Accuracy'], 2/3)
        self.assertAlmostEqual(Metrics['Precision'], 0.75)
        self.assertAlmostEqual(Metrics['Recall'], 0.75)
        self.assertAlmostEqual(Metrics['F1'], 2/3)

if __name__ == '__main__':
    unittest.main()
