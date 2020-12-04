from abc import ABC
from collections import defaultdict
import os
from typing import Mapping, Tuple

import numpy as np
from tqdm import tqdm, trange
import torch
from torch import optim
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from joblib import Parallel, delayed
import sklearn.linear_model as lm


class PredictionsSmoother(ABC):
    def __init__(self, tag='', n_neighbours=51):
        self.tag = tag
        self._n_neighbours = n_neighbours

    def __call__(
            self,
            predictions: np.array,
            points: np.array,
            predictions_variants: Mapping
    ) -> Tuple[np.array, Mapping]:

        n_omp_threads = int(os.environ.get('OMP_NUM_THREADS', 1))
        nn_distances, nn_indexes = cKDTree(points, leafsize=100) \
            .query(points, k=self._n_neighbours, n_jobs=n_omp_threads)

        smoothed_predictions = self.perform_smoothing(
            predictions, points, predictions_variants, nn_distances, nn_indexes)
        return smoothed_predictions

    def perform_smoothing(
            self,
            predictions: np.array,
            points: np.array,
            predictions_variants: Mapping,
            nn_distances: np.array,
            nn_indexes: np.array
    ) -> np.array:

        raise NotImplemented()


class OptimizationBasedSmoother(PredictionsSmoother):
    def __init__(self, regularizer_alpha=0.01, tag='', n_neighbours=51):
        super().__init__(tag, n_neighbours)
        self._regularizer_alpha = regularizer_alpha

    def smoothing_loss(self, *args, **kwargs):
        raise NotImplemented()

    def perform_smoothing(self, predictions, points, predictions_variants, nn_distances, nn_indexes):
        init_predictions_th = torch.Tensor(predictions)
        predictions_th = torch.ones(predictions.shape)
        predictions_th.requires_grad_()

        optimizer = optim.SGD([predictions_th], lr=0.001, momentum=0.9)
        t = trange(300, desc='Optimization', leave=True)
        for i in t:
            optimizer.zero_grad()
            loss = self.smoothing_loss(predictions_th, init_predictions_th, nn_indexes, self._regularizer_alpha)
            loss.backward()
            optimizer.step()
            s = 'Optimization: step #{0:}, loss: {1:3.1f}'.format(i, loss.item())
            t.set_description(s)
            t.refresh()

        return predictions_th.detach().numpy()


class L2Smoother(OptimizationBasedSmoother):
    def __init__(self, regularizer_alpha=0.01):
        super().__init__(regularizer_alpha, tag='l2')

    def smoothing_loss(self, predictions, init_predictions, nn_indexes, alpha):
        data_fidelity_term = (predictions - init_predictions) ** 2
        regularization_term = torch.sum(
            (predictions[nn_indexes[:, 1:]] -
             predictions.reshape((len(predictions), 1))) ** 2,
            dim=1)
        return torch.sum(data_fidelity_term) + alpha * torch.sum(regularization_term)


class TotalVariationSmoother(OptimizationBasedSmoother):
    def __init__(self, regularizer_alpha=0.01):
        super().__init__(regularizer_alpha, tag='tv')

    def smoothing_loss(self, predictions, init_predictions, nn_indexes, alpha):
        data_fidelity_term = (predictions - init_predictions) ** 2
        regularization_term = torch.sum(
            torch.abs(
                predictions[nn_indexes[:, 1:]] -
                predictions.reshape((len(predictions), 1))),
            dim=1)
        return torch.sum(data_fidelity_term) + alpha * torch.sum(regularization_term)


class RobustLocalLinearFit(PredictionsSmoother):
    def __init__(self, estimator, n_jobs=1, n_neighbours=51):
        super().__init__(tag='linreg', n_neighbours=n_neighbours)
        self._n_jobs = n_jobs
        self._estimator = estimator

    def perform_smoothing(self, predictions, points, predictions_variants, nn_distances, nn_indexes):

        def make_xy(point_index, points, nn_indexes, predictions_variants):
            X, y = [], []
            for neighbour_index in nn_indexes[point_index]:
                X.extend([points[neighbour_index]] * len(predictions_variants[neighbour_index]))
                y.extend(predictions_variants[neighbour_index])
                for y_value in predictions_variants[neighbour_index]:
                    X.append(points[neighbour_index])
                    y.append(y_value)

            return np.array(X), np.array(y), np.unique(X, axis=0, return_index=True)[1]

        def data_maker(points, nn_indexes, predictions_variants):
            for point_index in range(len(points)):
                X, y, uniq_indexes = make_xy(point_index, points, nn_indexes, predictions_variants)
                yield point_index, X, y, uniq_indexes

        def local_linear_fit_with_pipe(X, y):
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=2)),
                ('feat', PolynomialFeatures(2)),
                ('reg', lm.HuberRegressor(epsilon=4., alpha=1., max_iter=1000))])
            try:
                y_pred = pipe.fit(X, y).predict(X)
            except ValueError:
                y_pred = None
            return y_pred

        def local_linear_fit(X, y, estimator):
            X_trans = PCA(n_components=2).fit_transform(X)
            X_trans = PolynomialFeatures(2).fit_transform(X_trans)
            try:
                y_pred = estimator.fit(X_trans, y).predict(X_trans)
            except ValueError:
                y_pred = None
            return y_pred

        parallel = Parallel(n_jobs=self._n_jobs, backend='loky', verbose=100)
        #       delayed_iterable = (delayed(local_linear_fit)(X, y, deepcopy(self._estimator))
        #                           for point_index, X, y, uniq_indexes in data_maker(points, nn_indexes, predictions_variants))
        delayed_iterable = (delayed(local_linear_fit_with_pipe)(X, y)
                            for point_index, X, y, uniq_indexes in data_maker(points, nn_indexes, predictions_variants))
        refined_predictions = parallel(delayed_iterable)

        refined_predictions_variants = defaultdict(list)
        for refined_prediction, (point_index, X, y, uniq_indexes) in tqdm(
                zip(refined_predictions, data_maker(points, nn_indexes, predictions_variants))):
            if None is refined_prediction:
                continue
            for ui, nn_index in enumerate(zip(uniq_indexes, nn_indexes[point_index])):
                refined_predictions_variants[nn_index].append(refined_prediction[ui])

        refined_combined_predictions = np.zeros_like(predictions)
        for idx, values in refined_predictions_variants.items():
            refined_combined_predictions[idx] = np.mean(values)

        return refined_combined_predictions
