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
        self.pipeline = pipelines.baseline_net(max_epochs=10, dropout=0.2, lr=1e-1)
        self.training_sample_path = os.path.join("tests", "training_sample_100.csv")

    def test_fit(self):
        X, y = load_from_csv(self.training_sample_path)
        self.pipeline.fit(X, y)
        train_losses = [(h['epoch'], h['train_loss']) for h in self.pipeline._final_estimator.history]
        self.assertEqual(
            train_losses,
            [(1, 0.19911019504070282),
             (2, 0.07006944715976715),
             (3, 0.04542629420757294),
             (4, 0.03575495630502701),
             (5, 0.03279704973101616),
             (6, 0.025873012840747833),
             (7, 0.023428773507475853),
             (8, 0.023023175075650215),
             (9, 0.015065216459333897),
             (10, 0.01120317354798317)]
        )

    def test_internal_cross_validate(self):
        self.pipeline = pipelines.baseline_net(max_epochs=10, train_split=CVSplit(0.2, random_state=0))
        X, y = load_from_csv(self.training_sample_path)
        self.pipeline.fit(X, y)
        losses = [(h['epoch'], h['train_loss'], h['valid_loss']) for h in self.pipeline._final_estimator.history]
        self.assertEqual(
            losses,
            [(1, 0.10234718769788742, 0.02837774157524109), (2, 0.0399295836687088, 0.024141712114214897),
             (3, 0.031169436872005463, 0.021233337000012398), (4, 0.025440266355872154, 0.01903749629855156),
             (5, 0.021385645493865013, 0.017291951924562454), (6, 0.01837981678545475, 0.015881769359111786),
             (7, 0.016166657209396362, 0.014710157178342342), (8, 0.0143850427120924, 0.013715007342398167),
             (9, 0.012921981513500214, 0.012865738943219185), (10, 0.011702113784849644, 0.012164301238954067)]
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
