import weakref

from pytorch_lightning.core.lightning import LightningModule


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def __init__(self, pl_module: LightningModule, dataset_name: str = ""):
        self.dataset_name = dataset_name
        self.pl_module = weakref.ref(pl_module) if pl_module is not None else None

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

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        self._evaluators = evaluators
        pl_module = None
        dataset_name = ""
        if len(self._evaluators) > 0:
            dataset_name = self._evaluators[0].dataset_name
            pl_module = self._evaluators[0].pl_module
            for i in range(1, len(self._evaluators)):
                assert pl_module == self._evaluators[i].pl_module
                assert dataset_name == self._evaluators[i].dataset_name
        super().__init__(pl_module, dataset_name)

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
