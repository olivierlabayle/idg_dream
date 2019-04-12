import unittest
import torch
import os
import numpy as np

from skorch.dataset import CVSplit
from sklearn.metrics import mean_squared_error

from idg_dream import pipelines
from idg_dream.utils import load_from_csv


class TestBaselineNetPipeline(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.pipeline = pipelines.baseline_net(max_epochs=10, lr=1e-5)
        self.training_sample_path = os.path.join("tests", "training_sample_100.csv")

    def test_fit(self):
        X, y = load_from_csv(self.training_sample_path)
        self.pipeline.fit(X, y)
        train_losses = [(h['epoch'], h['train_loss']) for h in self.pipeline._final_estimator.history]
        self.assertEqual(
            train_losses,
            [(1, 0.12810704112052917),
             (2, 0.048660922795534134),
             (3, 0.04263147711753845),
             (4, 0.0328749381005764),
             (5, 0.021398983895778656),
             (6, 0.024104291573166847),
             (7, 0.01671551913022995),
             (8, 0.018973413854837418),
             (9, 0.017443161457777023),
             (10, 0.013933315873146057)]
        )

    def test_internal_cross_validate(self):
        self.pipeline = pipelines.baseline_net(max_epochs=10, train_split=CVSplit(0.2, random_state=0))
        X, y = load_from_csv(self.training_sample_path)
        self.pipeline.fit(X, y)
        losses = [(h['epoch'], h['train_loss'], h['valid_loss']) for h in self.pipeline._final_estimator.history]
        self.assertEqual(
            losses,
            [(1, 0.10234718769788742, 0.02837774157524109),
             (2, 0.0399295836687088, 0.024141710251569748),
             (3, 0.03116944059729576, 0.021233338862657547),
             (4, 0.025440268218517303, 0.01903749629855156),
             (5, 0.021385645493865013, 0.017291953787207603),
             (6, 0.01837981678545475, 0.015881771221756935),
             (7, 0.016166657209396362, 0.014710159972310066),
             (8, 0.01438504084944725, 0.013715006411075592),
             (9, 0.012921982444822788, 0.012865740805864334),
             (10, 0.011702114716172218, 0.012164303101599216)]
        )


class TestLinearRegressionPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.pipeline = pipelines.linear_regression()
        self.training_sample_path = os.path.join("tests", "training_sample_100.csv")

    def test_fit(self):
        X, y = load_from_csv(self.training_sample_path)
        self.pipeline.fit(X, y)
        y_pred = self.pipeline.predict(X)
        mse = mean_squared_error(y, y_pred)
        self.assertEqual(mse, 7.448570001894196e-14)
        np.testing.assert_allclose(self.pipeline._final_estimator.intercept_, np.array([7.069288e-06]))


class TestBiLSTMFingerprintPepeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.pipeline = pipelines.bilstm_fingerprint(lr=0.1, max_epochs=5)
        self.training_sample_path = os.path.join("tests", "training_sample_100.csv")

    def test_fit(self):
        X, y = load_from_csv(self.training_sample_path)
        self.pipeline.fit(X, y)
        train_losses = [(h['epoch'], h['train_loss']) for h in self.pipeline._final_estimator.history]
        self.assertEqual(
            train_losses,
            [(1, 0.011257863603532314),
             (2, 0.007060164585709572),
             (3, 0.004600393120199442),
             (4, 0.0031647691503167152),
             (5, 0.0023271518293768167)]
        )
