import os
import glob
import time
import tqdm
import tensorboardX
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import hawtorch
import hawtorch.nn as hnn
from hawtorch.io import logger
from hawtorch.utils import DelayedKeyboardInterrupt, summary


class Trainer(object):
    """Base trainer class. 
    """

    def __init__(self, 
                 model, 
                 optimizer, 
                 lr_scheduler, 
                 objective, 
                 device,
                 dataloaders,
                 logger,
                 metrics=[],
                 input_shape=None,
                 workspace_path=None, 
                 use_checkpoint="latest",
                 restart=False,
                 max_keep_ckpt=10,
                 eval_set="val",
                 test_set="test",
                 eval_interval=1,
                 save_interval=1,
                 report_step_interval=300,
                 use_parallel=False,
                 use_tqdm=True,
                 use_tensorboardX=True,
                 weight_init_function=None,
                 ):
                 
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.objective = objective
        self.device = device
        self.dataloaders = dataloaders
        self.metrics = metrics
        self.log = logger
        self.workspace_path = workspace_path
        self.use_checkpoint = use_checkpoint
        self.max_keep_ckpt = max_keep_ckpt
        self.restart = restart
        self.eval_set = eval_set
        self.test_set = test_set
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.report_step_interval = report_step_interval
        self.use_parallel = use_parallel
        self.use_tqdm = use_tqdm
        self.use_tensorboardX = use_tensorboardX

        self.model.to(self.device)
        if input_shape is not None:
            summary(self.model, input_shape, logger=self.log)
        if self.use_parallel:
            self.model = nn.DataParallel(self.model)
        if weight_init_function is not None:
            self.model.apply(weight_init_function)

        self.log.info('Number of model parameters: {}'.format(sum([p.numel() for p in model.parameters() if p.requires_grad])))

        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "StepLoss": [],
            "EpochLoss": [],
            "EvalResults": [],
            "Checkpoints": [],
            "BestResult": None,
            }

        if self.workspace_path is not None:
            os.makedirs(self.workspace_path, exist_ok=True)
            if self.use_checkpoint == "latest":
                self.log.info("Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "scratch":
                self.log.info("Train from scratch")
            elif self.use_checkpoint == "best":
                self.log.info("Loading best checkpoint ...")
                model_name = type(self.model).__name__
                ckpt_path = os.path.join(self.workspace_path, 'checkpoints')
                best_path = f"{ckpt_path}/{model_name}_best.pth.tar"
                self.load_checkpoint(best_path)
            else: # path to ckpt
                self.log.info(f"Loading checkpoint {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def train(self, max_epochs):
        """
        do the training process for max_epochs.
        """
        if self.use_tensorboardX:
            time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            logdir = os.path.join(self.workspace_path, "run", time_stamp)
            self.writer = tensorboardX.SummaryWriter(logdir)

        
        for epoch in range(self.epoch, max_epochs+1):
            self.epoch = epoch
            
            self.train_one_epoch()

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(self.eval_set)

            if self.workspace_path is not None and self.epoch % self.save_interval == 0:
                self.save_checkpoint()

        if self.use_tensorboardX:
            self.writer.close()

        self.log.info("Finished Training.")

    def evaluate(self, eval_set=None):
        """
        final evaluate at the best epoch.
        """
        self.log.info(f"Evaluate at the best epoch on {eval_set} set...")
        model_name = type(self.model).__name__
        ckpt_path = os.path.join(self.workspace_path, 'checkpoints')
        best_path = f"{ckpt_path}/{model_name}_best.pth.tar"
        if not os.path.exists(best_path):
            self.log.error(f"Best checkpoint not found! {best_path}")
            raise FileNotFoundError
        self.load_checkpoint(best_path)
        # turn off logging to tensorboardX
        use_tensorboardX_old = self.use_tensorboardX
        self.use_tensorboardX = False
        if eval_set is None: 
            eval_set = self.eval_set
        self.evaluate_one_epoch(eval_set)
        self.use_tensorboardX = use_tensorboardX_old

    def get_time(self):
        if torch.cuda.is_available(): 
            torch.cuda.synchronize()
        return time.time()

    def train_one_epoch(self):
        self.log.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']} ...")

        for metric in self.metrics:
            metric.clear()

        self.model.train()
        epoch_start_time = self.get_time()

        outputs, As = self.model(self.dataloaders["features"]) # [1, N, nCls]
        outputs = outputs.squeeze().permute(1,0)

        
        mask = self.dataloaders["train_idx"]

        outputs = outputs[mask]
        truths = self.dataloaders["labels"][mask]

        loss = self.objective(outputs, truths, self.dataloaders["features"], As)


        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        for metric in self.metrics:
            metric.update(outputs, truths)
            if self.use_tensorboardX:
                metric.write(self.writer, self.epoch, prefix="train")
                    
        if self.use_tensorboardX:
            self.writer.add_scalar("train/loss", loss.item(), self.epoch)

        for metric in self.metrics:
            self.log.log1(metric.report())
            metric.clear()

        epoch_end_time = self.get_time()
        self.stats["EpochLoss"].append(loss.item())

        self.log.log(f"==> Finished Epoch {self.epoch}, loss={loss.item():.4f}, time={epoch_end_time-epoch_start_time:.4f}")


    def evaluate_one_epoch(self, eval_set):
        self.log.log(f"++> Evaluate at epoch {self.epoch} ...")

        self.model.eval()
        for metric in self.metrics:
            metric.clear()

        epoch_start_time = self.get_time()
        with torch.no_grad():
            start_time = self.get_time()

            outputs, As = self.model(self.dataloaders["features"]) # [N, nCls]
            outputs = outputs.squeeze().permute(1,0)

            mask = self.dataloaders["test_idx"]
            outputs = outputs[mask]
            truths = self.dataloaders["labels"][mask]
            loss = self.objective(outputs, truths, self.dataloaders["features"], As)

            for metric in self.metrics:
                metric.update(outputs, truths)

            self.stats["EvalResults"].append(self.metrics[0].measure())

            for metric in self.metrics:
                self.log.log1(metric.report())
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()
            
            if self.use_tensorboardX:
                self.writer.add_scalar("evaluate/loss", loss.item(), self.epoch)

        epoch_end_time = self.get_time()
        self.log.log(f"++> Evaluate Finished. time={epoch_end_time-epoch_start_time:.4f}")

    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""
        with DelayedKeyboardInterrupt():
            model_name = type(self.model).__name__
            ckpt_path = os.path.join(self.workspace_path, 'checkpoints')
            file_path = f"{ckpt_path}/{model_name}_ep{self.epoch:04d}.pth.tar"
            best_path = f"{ckpt_path}/{model_name}_best.pth.tar"
            os.makedirs(ckpt_path, exist_ok=True)

            self.stats["Checkpoints"].append(file_path)

            if len(self.stats["Checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["Checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
                    self.log.info(f"Removed old checkpoint {old_ckpt}")

            state = {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'model_name': model_name,
                'model': self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'stats' : self.stats,
            }

            if self.stats["BestResult"] is None or self.metrics[0].better(self.stats["EvalResults"][-1], self.stats["BestResult"]):
                self.stats["BestResult"] = self.stats["EvalResults"][-1]
                torch.save(state, best_path)
                self.log.info(f"Saved Best checkpoint.")
            
            torch.save(state, file_path)
            self.log.info(f"Saved checkpoint {self.epoch} successfully.")


    def load_checkpoint(self, checkpoint=None):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the modelwork at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        model_name = type(self.model).__name__
        
        ckpt_path = os.path.join(self.workspace_path, 'checkpoints')
        
        if checkpoint is None:
            # Load most recent checkpoint            
            checkpoint_list = sorted(glob.glob(f'{ckpt_path}/{model_name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                self.log.info("No matching checkpoint found, train from zero.")
                return False
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = f'{ckpt_path}/{model_name}_ep{checkpoint:04d}.pth.tar'
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            self.log.error("load_checkpoint: Invalid argument")
            raise TypeError

        checkpoint_dict = torch.load(checkpoint_path)

        #assert model_name == checkpoint_dict['model_name'], 'network is not of correct type.'

        self.model.load_state_dict(checkpoint_dict['model'])
        if not self.restart:
            self.log.info("Loading epoch and other status...")
            self.epoch = checkpoint_dict['epoch'] + 1
            self.global_step = checkpoint_dict['global_step']
            self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
            self.lr_scheduler.last_epoch = checkpoint_dict['epoch'] 
            self.stats = checkpoint_dict['stats']
        else:
            self.log.info("Only loading model parameters.")
        
        self.log.info("Checkpoint Loaded Successfully.")
        return True


