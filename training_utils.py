import os
import torch as th
from torcheval.metrics import BinaryAUROC, MeanSquaredError
import tqdm
import h5py
import gc
import yaml
from torch.profiler import profile, record_function, ProfilerActivity
from pm import re_namedtuple

class Trainer:
    @staticmethod
    def _split_batch(batch):
        if not isinstance(batch, (list, tuple)):
            raise TypeError(f"Expected batch to be a list or tuple but got {type(batch)!r}")

        if len(batch) == 5:
            X0, X1, X2, Y1, Y2 = batch
            origin = None
        elif len(batch) == 6:
            X0, X1, X2, Y1, Y2, origin = batch
        else:
            raise ValueError(f"Unexpected batch size {len(batch)}. Expected 5 (train/val) or 6 (test) elements.")

        return X0, X1, X2, Y1, Y2, origin

    class Metric:
        def __init__(self, name):
            self.name = name
            self.metric1 = BinaryAUROC(device='cuda')
            self.metric2 = MeanSquaredError(device='cuda')
            self.n = 0.001
            self.loss = 0
            self.weight = None
            self.uncertainties = []

        def reset(self):
            self.n = 0.001
            self.loss = 0
            self.metric1.reset()
            self.metric2.reset()
            self.weight = None
            self.uncertainties = []

        def update(self, output, target, loss_item, weight=None):
            self.metric1.update(output[0][:].detach().ravel(), target[0][:].detach().ravel())
            self.metric2.update(output[1][:].detach().ravel(), target[1][:].detach().ravel())
            if len(output) > 2:
                self.uncertainties.append(output[2].detach().cpu())
            self.n += len(output[0])
            self.loss += loss_item * len(output[0])
            self.weight = weight

        def get_loss(self):
            return self.loss / self.n

        def get_desc(self):
            desc = f"{self.name} loss=({self.loss / self.n:.3f}), auc {self.metric1.compute().item():.3f}, mse {self.metric2.compute().item():.2f}"
            if self.uncertainties:
                log_sigma = th.cat(self.uncertainties, dim=0)
                desc += f", sigma={log_sigma.exp().mean().item():.2f}"
            return desc

    def __init__(self, model_wrapper, data_config, model_config, name, condor=False, epoch_min=30):
        self.model_wrapper = model_wrapper
        self.data_config = data_config
        self.model_config = model_config
        self.name = name
        self.condor = condor
        self.epoch_min = epoch_min

    def train(self, train_loader, val_loader=None, num_epoch=100):
        train_metric = self.Metric("train")
        val_metric = self.Metric("val")
        best_loss = 30
        early_stop = 0
        check = 0
        sample = train_loader.dataset[0]
        X0, X1, X2, Y1, Y2, _ = self._split_batch(sample)
        print("X0", X0.shape, "X1", X1.shape, "X2", X2.shape)
        inputs0 = th.empty(self.model_wrapper.batch_size, X0.shape[0], X0.shape[1], device='cuda')
        inputs1 = th.empty(self.model_wrapper.batch_size, X1.shape[0], X1.shape[1], device='cuda')
        inputs2 = th.empty(self.model_wrapper.batch_size, X2.shape[0], X2.shape[1], device='cuda')
        targets1 = th.empty(self.model_wrapper.batch_size, Y1.shape[0], device='cuda')
        targets2 = th.empty(self.model_wrapper.batch_size, Y2.shape[0], device='cuda')

        for epoch in range(num_epoch):
            train_metric.reset()
            th.set_grad_enabled(True)
            if not self.condor:
                bar = tqdm.tqdm(total=len(train_loader), nrows=2, leave=val_loader is None)
            self.model_wrapper.net.train()
            tick = 0
            for batch in train_loader:
                X0, X1, X2, Y1, Y2, _ = self._split_batch(batch)
                inputs0.copy_(X0, non_blocking=True)
                inputs1.copy_(X1, non_blocking=True)
                inputs2.copy_(X2, non_blocking=True)
                targets1.copy_(Y1, non_blocking=True)
                targets2.copy_(Y2, non_blocking=True)
                output = self.model_wrapper.net(inputs0, inputs1, inputs2)
                if check == 0:
                    print('targets2', targets2[:5])
                    check = 1
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                        with record_function("model_inference"):
                            output = self.model_wrapper.net(inputs0, inputs1, inputs2)
                    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

                self.model_wrapper.optim.zero_grad()

                loss = self.model_wrapper.get_loss(output, [targets1, targets2], epoch)
                if tick % 100 == 0:
                    train_metric.reset()
                Y1 = Y1.to("cuda", non_blocking=True)
                Y2 = Y2.to("cuda", non_blocking=True)
                train_metric.update(output, [Y1, Y2], loss.item())

                if train_metric.metric2.compute().item() > 1000000:
                    print("X1", X1.sum(1))
                    print("X2", X2.cpu())
                    print("out2", output[0], output[1])
                    break

                assert not th.isnan(output[0]).any(), "output0 in loss"
                assert not th.isnan(output[1]).any(), "output1 in loss"
                assert not th.isnan(output[2]).any(), "output1 in loss"
                assert not th.isnan(loss).any(), "NaN in loss"
                try:
                    loss.backward(retain_graph=True)
                    self.model_wrapper.optim.step()
                except:
                    print("tick", tick)
                    print("loss", loss)
                    self.model_wrapper.optim.zero_grad()
                    loss.backward(retain_graph=True)
                    self.model_wrapper.optim.step()

                if not self.condor:
                    bar.set_description(f"epoch{epoch:4d} : {train_metric.get_desc()} ")
                    bar.update()
                del X0, X1, X2, Y1, Y2, loss, output
                th.cuda.empty_cache()
                tick += 1

            if self.condor:
                print(f"epoch{epoch:4d} : {train_metric.get_desc()} ")
            gc.collect()
            if val_loader is not None:
                val_metric.reset()
                if not self.condor:
                    bar = tqdm.tqdm(total=len(val_loader), nrows=2)
                self.model_wrapper.net.eval()
                th.set_grad_enabled(False)
                tick = 0

                for batch in val_loader:
                    X0, X1, X2, Y1, Y2, _ = self._split_batch(batch)
                    X0 = X0.to("cuda", non_blocking=True)
                    X1 = X1.to("cuda", non_blocking=True)
                    X2 = X2.to("cuda", non_blocking=True)
                    Y1 = Y1.to("cuda", non_blocking=True)
                    if check == 1:
                        print('Y2', Y2[:5])
                        check = 2
                    Y2 = Y2.to("cuda", non_blocking=True)
                    output = self.model_wrapper.net(X0, X1, X2)

                    loss = self.model_wrapper.get_loss(output, [Y1, Y2])
                    lossitem = loss.item()
                    val_metric.update(output, [Y1, Y2], lossitem)
                    if lossitem > 10000:
                        print("tick", tick)
                        print('X2', X2[:])
                        break

                    if not self.condor:
                        bar.set_description(f"epoch{epoch:4d} : {train_metric.get_desc()} {val_metric.get_desc()} ")
                        bar.update()
                    del X0, X1, X2, Y1, Y2, loss, output
                    th.cuda.empty_cache()
                    tick += 1

                if self.condor:
                    print(f"epoch{epoch:4d} : {train_metric.get_desc()} {val_metric.get_desc()} ")
                if val_metric.get_loss() < best_loss or epoch == 30:
                    early_stop = 0
                    best_loss = val_metric.get_loss()
                    self.save_checkpoint(epoch, best_loss)
                if epoch > self.epoch_min and early_stop > 8:
                    break
                else:
                    early_stop += 1

    def test(self, test_loader):
        th.set_grad_enabled(False)
        self.model_wrapper.net.eval()
        with h5py.File(f'save/{self.name}_best.h5py', 'w') as hf:
            sample = test_loader.dataset[0]
            X0, X1, X2, Y1, Y2, origin = self._split_batch(sample)
            if origin is None:
                raise ValueError("Test loader batch is expected to include origin information.")
            test_Y1 = hf.create_dataset("test_Y1", shape=(len(test_loader.dataset), Y1.shape[0]), dtype='float32')
            test_Y2 = hf.create_dataset("test_Y2", shape=(len(test_loader.dataset), Y2.shape[0]), dtype='float32')
            test_O = hf.create_dataset("test_O", shape=(len(test_loader.dataset), origin.shape[0]), dtype='int32')
            test_p1 = hf.create_dataset("test_p1", shape=(len(test_loader.dataset), Y1.shape[0]), dtype='float32')
            test_p2 = hf.create_dataset("test_p2", shape=(len(test_loader.dataset), Y2.shape[0]), dtype='float32')
            test_e2 = hf.create_dataset("test_e2", shape=(len(test_loader.dataset), Y2.shape[0]), dtype='float32')
            if not self.condor:
                bar = tqdm.tqdm(total=len(test_loader), nrows=2)
            test_metric = self.Metric("test")
            n_t = 0
            for batch in test_loader:
                X0, X1, X2, Y1, Y2, origin = self._split_batch(batch)
                if origin is None:
                    raise ValueError("Test loader batch is expected to include origin information.")
                test_O[n_t:n_t + len(X1)] = origin
                test_Y1[n_t:n_t + len(X1)] = Y1
                test_Y2[n_t:n_t + len(X1)] = Y2[:]
                X0 = X0.to("cuda")
                X1 = X1.to("cuda")
                X2 = X2.to("cuda")
                Y1 = Y1.to("cuda")
                Y2 = Y2.to("cuda")
                output = self.model_wrapper.net(X0, X1, X2)
                loss = self.model_wrapper.get_loss(output, [Y1, Y2])
                test_metric.update(output, [Y1, Y2], loss.item())
                test_p1[n_t:n_t + len(X1)] = th.nn.functional.softmax(output[0], dim=1).cpu().numpy()
                test_p2[n_t:n_t + len(X1)] = output[1].cpu().numpy()
                test_e2[n_t:n_t + len(X1)] = output[2].cpu().numpy()
                n_t += len(X1)
                if not self.condor:
                    bar.set_description(f"{test_metric.get_desc()}")
                    bar.update()
            if self.condor:
                print(f"{test_metric.get_desc()}")

            cfg = re_namedtuple(yaml.safe_load(self.data_config)).input0
            hf.attrs['cand'] = ",".join(self.model_wrapper.cand)
            hf.attrs['target'] = ",".join(cfg.target)
            hf.attrs['target_scale'] = ",".join([str(t) for t in cfg.target_scale])
        print(f'save/{self.name}_best.h5py')

    def save_checkpoint(self, epoch, best_loss):
        if not os.path.isdir("save"):
            os.makedirs("save")
        th.save({
            'net': self.model_wrapper.net.state_dict(),
            'optimizer': self.model_wrapper.optim.state_dict(),
            'best_loss': best_loss,
            'epoch': epoch,
            'data_config': self.data_config,
            'model_config': self.model_config,
        }, f'save/ckpt-{self.name}-best.pth')

    @staticmethod
    def load_checkpoint(name, model_wrapper=None):
        """Loads checkpoint. If model_wrapper is provided, it loads the network state dict."""
        tl = th.load(f'save/ckpt-{name}-best.pth', map_location=th.device('cpu'))
        if model_wrapper is not None:
            model_wrapper.net.load_state_dict(tl['net'])
        print("best loss", tl['best_loss'])
        return tl['data_config'], tl['model_config']

class ModelWrapper:
    def __init__(self, name, data_config, model_config, condor=False):
        self.data_config = data_config
        self.model_config = model_config
        self.name = name
        self.condor = condor
        cfg = re_namedtuple(yaml.safe_load(self.data_config)).input0
        self.batch_size = cfg.batch_size
        self.cand = cfg.cand
        self.net = None
        self.optim = None

    def set_net(self, net):
        self.net = net
        self.optim = th.optim.AdamW(self.net.parameters(), lr=0.001)
        self.loss_ce = th.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loss_mse = th.nn.MSELoss()
        self.loss_sml1 = th.nn.SmoothL1Loss(beta=0.1)

    def get_loss(self, output, target, epoch=3):
        loss_cls = self.loss_ce(output[0], target[0])

        # Clamp log_sigma for stability. The range can be a tunable hyperparameter.
        log_sigma = th.clamp(output[2], min=-5.0, max=5.0)

        # Detaching log_sigma for the first few epochs prevents the uncertainty head
        # from destabilizing the main regression task while it's still learning.
        if epoch < 2:
            log_sigma = th.zeros_like(log_sigma).detach()

        # Aleatoric uncertainty loss for regression
        # Loss = 0.5 * exp(-s) * (y - y_hat)^2 + 0.5 * s
        # where s = log(sigma^2) = 2 * log(sigma)
        loss_reg = 0.5 * th.exp(-2 * log_sigma) * ((target[1] - output[1]) ** 2) + log_sigma
        loss_reg = loss_reg.mean()
        return loss_cls + loss_reg
