import logging
from datetime import timedelta
from time import perf_counter

from pytorch_lightning import Callback

log = logging.getLogger(__name__)


class FitDurationCallback(Callback):

    def on_fit_start(self, trainer, pl_module):
        self.start_fit_time = perf_counter()

    def on_fit_end(self, trainer, pl_module):
        total_time = perf_counter() - self.start_fit_time
        log.info(f"Total trainer.fit() duration: {timedelta(seconds=int(total_time))}")
