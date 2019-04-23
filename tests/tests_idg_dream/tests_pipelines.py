import unittest
import torch
import os
import numpy as np

from skorch.dataset import CVSplit
from sklearn.metrics import mean_squared_error

from idg_dream import pipelines
from idg_dream.pipelines import GraphBiLSTMFactory, NNFactory
from idg_dream.utils import load_from_csv


class AbstractDeepPipelineTester:
    training_sample_path = os.path.join("tests", "training_sample_100.csv")

    def setUp(self):
        torch.manual_seed(0)

    def test_fit(self):
        X, y = load_from_csv(self.training_sample_path)
        self.pipeline.fit(X, y)
        train_losses = [(h['epoch'], h['train_loss']) for h in self.pipeline._final_estimator.history]
        for i in range(len(train_losses) - 1):
            self.assertGreater(train_losses[i][1], train_losses[i + 1][1])


class TestBaselineNetPipeline(unittest.TestCase, AbstractDeepPipelineTester):
    def setUp(self):
        super().setUp()
        self.pipeline = pipelines.BaselineNetFactory()(max_epochs=10, lr=1e-3)


class TestLinearRegressionPipeline(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.pipeline = pipelines.LinearRegressionFactory()()
        self.training_sample_path = os.path.join("tests", "training_sample_100.csv")

    def test_fit(self):
        X, y = load_from_csv(self.training_sample_path)
        self.pipeline.fit(X, y)
        y_pred = self.pipeline.predict(X)
        mse = mean_squared_error(y, y_pred)
        self.assertLessEqual(mse, 1)


class TestBiLSTMFingerprintPipeline(unittest.TestCase, AbstractDeepPipelineTester):
    def setUp(self):
        super().setUp()
        self.pipeline = pipelines.BiLSTMFingerprintFactory()(lr=0.1, max_epochs=5)


class TestGraphBiLSTMPipeline(unittest.TestCase, AbstractDeepPipelineTester):
    def setUp(self):
        super().setUp()
        self.pipeline = GraphBiLSTMFactory()(lr=0.1, max_epochs=5)

class TestProteinBasedKNN(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.pipeline = NNFactory()()
        self.training_sample_path = os.path.join("tests", "training_sample_100.csv")

    def test_fit(self):
        X, y = load_from_csv(self.training_sample_path)
        self.pipeline.fit(X, y)
        y_pred = self.pipeline.predict(X)
        mse = mean_squared_error(y, y_pred)
        self.assertLessEqual(mse, 1)