import weakref
from typing import Optional, List

import torch.nn as nn
from torch.utils.data import Dataset


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def __init__(self, model: Optional[nn.Module], dataset: Optional[Dataset], dataset_name: Optional[str]):
        self.dataset_name = dataset_name
        self.model = weakref.ref(model) if model is not None else None  # to call forward method
        self.dataset = dataset

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators: List[DatasetEvaluator]):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        self._evaluators = evaluators
        model = None
        dataset = None
        dataset_name = None
        if len(self._evaluators) > 0:
            model = self._evaluators[0].model
            dataset = self._evaluators[0].dataset
            dataset_name = self._evaluators[0].dataset_name
            for i in range(1, len(self._evaluators)):
                assert model == self._evaluators[i].model
                assert dataset == self._evaluators[i].dataset
                assert dataset_name == self._evaluators[i].dataset_name
        super().__init__(model, dataset, dataset_name)

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = {'scalars': {}, 'images': {}}
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            # if is_main_process() and result is not None:
            if result is not None:
                for meta_key in results.keys():
                    for k, v in result[meta_key].items():
                        assert (
                                k not in results[meta_key]
                        ), f"Different evaluators produce results ({meta_key}) with the same key {k}"
                        results[meta_key][k] = v
        return results
