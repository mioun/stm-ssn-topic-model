from abc import ABC

import numpy as np
import torch
from torch import nn
import snntorch as snn
from tqdm import tqdm

from cuda_data_set import CUDADataset
from gpu import stdp_learner


class STMGPUrunner(ABC):

    def __init__(self, input_size: int,
                 neuron_nbr: int,
                 tau_pre: float = 15.0,
                 tau_post: float = 15.0,
                 beta=0.8,
                 threshold: float = 0.1,
                 learning_rate: float = 0.0001,
                 device="cuda"
                 ):

        self.learning_rate = learning_rate
        self.device = device
        self.w_min = 0
        self.w_max = 10
        self.neuron_nbr = neuron_nbr
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, neuron_nbr),
            snn.Leaky(beta=beta, init_hidden=True, inhibition=True, threshold=threshold)
        )
        nn.init.uniform_(self.net[0].weight.data, 0, 1)
        self.net.to(self.device)
        self.sdp_learner = stdp_learner.STDPLearner2(synapse=self.net[0], sn=self.net[1],
                                                     tau_pre=tau_pre, tau_post=tau_post,

                                                     f_post=(lambda x: x * x * 0.2))
        self.sdp_learner.to(self.device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.)

    def train(self, loader: CUDADataset, epoch_nbr, steps):
        for _ in tqdm(range(epoch_nbr)):
            for batch_idx in range(loader.number_of_batches):
                batch = loader.load_batch(batch_idx).to_dense()
                batch = batch.reshape(loader.batch_size, steps, 3, self.input_size)
                print(batch.shape)
                batch = torch.clamp(batch.sum(dim=2), 0, 1)
                batch_in = batch.transpose(0, 1)
                self.train_batch(batch_in, steps)

    def encode_docs(self, loader: CUDADataset, steps):
        encoding = []
        self.sdp_learner.disable()
        for batch_idx in range(loader.number_of_batches):
            batch = loader.load_batch(batch_idx).to_dense()
            batch = batch.reshape(loader.batch_size, steps, 3, self.input_size)
            batch = torch.clamp(batch.sum(dim=2), 0, 1)
            batch_in = batch.transpose(0, 1)
            encoding.append(self.encode_batch(batch_in, steps))
        return torch.concat(encoding).numpy()

    def train_batch(self, batch, time_steps):
        with torch.no_grad():
            for t in range(time_steps):
                batch_t = batch[t].to(self.device)
                self.optimizer.zero_grad()
                self.net(batch_t)
                self.sdp_learner.step(on_grad=True)
                self.optimizer.step()
                self.net[0].weight.data.clamp_(self.w_min, self.w_max)

    def encode_batch(self, batch, time_steps):
        with torch.no_grad():
            batch_res = []
            for t in range(time_steps):
                batch_t = batch[t].to(self.device)
                batch_res.append(self.net(batch_t).cpu())
            return torch.stack(batch_res).sum(0)

    def topics(self, features) -> list:
        topics = []
        for i in range(self.neuron_nbr):
            w_after_training = self.net[0].weight.data.cpu().numpy()[i]
            top20 = [(features[idx], w_after_training[idx]) for idx in np.argsort(w_after_training)[-10:]]
            top20 = sorted(top20, key=lambda x: x[1], reverse=True)
            print(top20)
            topics.append([word for word, weight in top20])
        return topics

    @staticmethod
    def post_norm(x):
        return x * x * 0.2
