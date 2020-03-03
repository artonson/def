import sys
import torch
import numpy as np
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter


class Epoch:

    def __init__(self, task, model, loss, metrics, stage_name, gradient_clipping=None, writer=None, save_each_batch=None, visualization=False, save_dir='/home/gbobrovskih/', device='cpu', verbose=True):
        self.task = task
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.clipping_value = gradient_clipping
        self.writer = writer
        self.save_each_batch = save_each_batch
        self.vis = visualization
        self.save_dir = save_dir
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        # for metric in self.metrics:
        #     metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)

        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader, epoch):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {}
        for metric in self.metrics:
            if isinstance(metric.__name__, list):
                for name in metric.__name__:
                    for class_ in range(self.model.classes):
                        metrics_meters[name+'_'+str(class_)] = AverageValueMeter()
            else:
                metrics_meters[metric.__name__] = AverageValueMeter()

        metrics_all = np.zeros((len(dataloader.dataset), len(metrics_meters.keys()), 2))
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for i, iter_ in enumerate(iterator):
                x, y = iter_
                batch_size = x.shape[0]
                x, y = x.to(self.device), y.to(self.device)

                loss, y_pred = self.batch_update(x, y)

                # add images to tb
                if i%self.save_each_batch == 0 and self.vis == True:
                    self.writer.add_image('{} input'.format(self.stage_name), x[0][0].unsqueeze(0), i+len(iterator)*epoch)
                    if self.task == 'segmentation':
                        self.writer.add_image('{} gt'.format(self.stage_name), y[0][0].unsqueeze(0)+0.5*y[0][1].unsqueeze(0), i+len(iterator)*epoch)
                        self.writer.add_image('{} pred'.format(self.stage_name),
                                              y_pred[0][0].unsqueeze(0) + 0.5 * y_pred[0][1].unsqueeze(0),
                                              i + len(iterator) * epoch)
                    elif self.task == 'regression':
                        self.writer.add_image('{} gt'.format(self.stage_name), y[0], i+len(iterator)*epoch)
                        self.writer.add_image('{} pred'.format(self.stage_name), y_pred[0], i + len(iterator) * epoch)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)

                # add loss to tb
                if self.writer is not None:
                    self.writer.add_scalar('{} loss {}'.format(self.stage_name, self.loss.__name__), loss_meter.mean, i+len(iterator)*epoch)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                if self.model.activation is not None:
                    y_pred_activated = self.model.activation(y_pred)
                else:
                    y_pred_activated = y_pred

                if self.task == 'segmentation':
                    y_pred_activated[y_pred_activated >= 0.5] = 1.0
                    y_pred_activated[y_pred_activated < 0.5] = 0.0

                # metrics_logs={}
                # for i_met, metric_fn in enumerate(self.metrics):
                #     metric_value = []
                #     for i_bs in range(batch_size):
                #         per_class_metric = []
                #         for i_cl in range(self.model.classes): # num of classes
                #             value = metric_fn.forward(y_pred_activated[i_bs][i_cl], y[i_bs][i_cl])
                #             per_class_metric.append(value)#.cpu().detach().numpy()
                #             for i_name in range(len(value)):
                #                 metrics_all[i_bs+i*batch_size][i_name][i_cl] = value[i_name]
                #
                #         metric_value.append(per_class_metric)
                #     metric_value = np.array(metric_value).reshape(batch_size, self.model.classes, len(metric_fn.__name__))
                #     for j, name in enumerate(metric_fn.__name__):
                #         for class_ in range(self.model.classes):
                #             metrics_logs[name+'_'+str(class_)] = metric_value[:, class_, j].mean()

                #metrics_logs = {k: v.mean for k, v in metrics_meters.items()}

                # if self.stage_name == 'valid':
                #     logs.update(metrics_logs)

                if self.verbose and i%self.save_each_batch == 0:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        #torch.save(metrics_all, '{}/all_metrics_{}'.format(self.save_dir, self.stage_name))
        return logs


class TrainEpoch(Epoch):

    def __init__(self, task, model, loss, metrics, optimizer, gradient_clipping=None, writer=None, save_each_batch=None, save_dir='/home/gbobrovskih/', device='cpu', verbose=True):
        super().__init__(
            task=task,
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            writer=writer,
            save_each_batch=save_each_batch,
            save_dir=save_dir,
            gradient_clipping=gradient_clipping, 
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer


    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        if self.clipping_value is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_value)
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, task, model, loss, metrics, writer=None, save_each_batch=None, save_dir='/home/gbobrovskih/', device='cpu', verbose=True):
        super().__init__(
            task=task,
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            writer=writer,
            save_each_batch=save_each_batch,
            save_dir=save_dir,
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction

class PredictEpoch(Epoch):

    def __init__(self, task, model, loss, metrics, writer=None, save_each_batch=None, save_dir='/home/gbobrovskih/', device='cpu', verbose=True):
        super().__init__(
            task=task,
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            writer=writer,
            save_each_batch=save_each_batch,
            save_dir=save_dir,
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        pass

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.predict(x)
            loss = self.loss(prediction, y)
        return loss, prediction