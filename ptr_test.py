from dataclasses import dataclass
from typing import Iterator, List, Optional
from pprint import pprint
from random import shuffle

import dataclasses

from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from convex_hull_dataset import create_dataset, get_dataloader
from convex_hull_dataset import ConvexHullDataset, ConvexHullSample
from ptr_network import PointerNet


@dataclass
class TrainingState:
    optimizer: Optimizer
    lr_scheduler: Optional[_LRScheduler]

    # number of loss.backwards() calls before optimizer.step() is called
    mini_batch_size: int

    training_loss: List[float]
    validation_loss: List[float]

    # gets validation after validation_rate minibatches
    validation_rate: int
    # saves a checkpoint after checkpoint_rate epochs
    checkpoint_rate: int

    epochs: int
    current_epoch: int

    @staticmethod
    def make_default(parameters: Iterator[nn.Parameter]) -> "TrainingState":
        optimizer = SGD(parameters, lr=0.01, momentum=0.7)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.75)
        return TrainingState(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            mini_batch_size=2,
            training_loss=[],
            validation_loss=[],
            validation_rate=1,
            checkpoint_rate=5,
            epochs=20,
            current_epoch=0,
        )


class Trainer:
    state: TrainingState
    model: PointerNet
    training_dataset: ConvexHullDataset
    validation_dataset: ConvexHullDataset
    name: str

    def __init__(
        self,
        model: PointerNet,
        training_dataset: ConvexHullDataset,
        validation_dataset: ConvexHullDataset,
        name: str = "pointer_net_trainer_2",
    ):
        self.model = model
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.state = TrainingState.make_default(model.parameters())
        self.name = name

    def run_epoch(self, dataset: ConvexHullDataset, is_training: bool = True) -> float:
        epoch_loss = 0.0
        if is_training:
            shuffle(dataset)
            print(f"training with {len(dataset)} samples")
        else:
            print(f"validating with {len(dataset)} samples")
        with torch.set_grad_enabled(is_training):
            for i, sample in tqdm(enumerate(self.training_dataset, 1)):
                output = self.model(
                    sample.points,
                    positions=sample.vertices,
                    teacher_forcing=True,
                )
                for loss in output.loss:
                    epoch_loss += loss.item()
                if is_training:
                    self.state.optimizer.zero_grad()
                    for loss in output.loss:
                        loss.backward(retain_graph=True)
                    if i % self.state.mini_batch_size == 0:
                        self.state.optimizer.step()
        print(f"epoch loss: loss: {epoch_loss:8.3f}")
        return epoch_loss

    def train(self):
        try:
            for epoch in range(1, self.state.epochs + 1):
                print(f"Starting epoch: {epoch}")
                epoch_loss = self.run_epoch(self.training_dataset, is_training=True)
                self.state.training_loss.append(epoch_loss)
                if epoch % self.state.validation_rate == 0:
                    # this is "smart training" (...)
                    # if our validation scores decrease, then step the lr_scheduler
                    if not self.validate():
                        print('stepping lr')
                        self.state.lr_scheduler.step()
                if epoch % self.state.checkpoint_rate == 0:
                    self.checkpoint(f"epoch_{epoch}")
        except KeyboardInterrupt:
            self.checkpoint(f"interrupt")

    def validate(self) -> bool:
        validation_loss = self.run_epoch(self.validation_dataset, is_training=False)
        losses = self.state.validation_loss
        losses.append(validation_loss)
        return len(losses) < 2 or losses[-1] >= losses[-2]

    def checkpoint(self, note: str):
        torch.save(self, f"{self.name}_{note}.pt")

def run_new_training():
    training_dataset = create_dataset(2 ** 16)
    validation_dataset = create_dataset(2 ** 8)
    hidden_d = 128
    hidden_v = 64
    model = PointerNet(
        encoder_args={"hidden_d": hidden_d},
        decoder_args={"hidden_d": hidden_d, "hidden_v": hidden_v},
    )

    trainer = Trainer(model, training_dataset, validation_dataset)
    trainer.train()


