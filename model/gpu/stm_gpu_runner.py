import os
import pickle
import random
from abc import ABC

import numpy as np
import torch
from torch import nn
import snntorch as snn
from tqdm import tqdm

from cuda_data_set import CUDADataset
from model.gpu import stdp_learner


class STMGPUrunner(ABC):

    def __init__(self, input_size: int,
                 neuron_nbr: int,
                 tau_pre: float = 15.0,
                 tau_post: float = 15.0,
                 beta=0.8,
                 threshold: float = 0.1,
                 learning_rate: float = 0.0001,
                 device="cuda",
                 squuze=3,
                 weights_data=None,
                 inhibition=None
                 ):

        self.squuze = squuze
        self.learning_rate = learning_rate
        self.device = device
        self.beta = beta
        self.threshold = threshold
        self.w_min = 0
        self.w_max = 10
        self.neuron_nbr = neuron_nbr
        self.input_size = input_size
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        if inhibition is None:
            self.inhibition = self.threshold
        else:
            self.inhibition = inhibition

        self.net = nn.Sequential(
            nn.Linear(input_size, neuron_nbr),
            snn.Leaky(beta=beta, init_hidden=True, inhibition=True, threshold=threshold, reset_mechanism='zero')
        )
        if weights_data is not None:
            torch_weights = torch.from_numpy(weights_data).float()
            self.net[0].weight = nn.Parameter(torch_weights)
            self.net[0].half()
        else:
            nn.init.uniform_(self.net[0].weight.data, 0, 1)
        self.net = self.net.half()
        self.net.to(self.device)
        self.sdp_learner = stdp_learner.STDPLearner2(synapse=self.net[0], sn=self.net[1],
                                                     tau_pre=tau_pre, tau_post=tau_post,

                                                     f_post=(lambda x: x * x))
        self.sdp_learner.to(self.device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.)

    def train(self, loader: CUDADataset, epoch_nbr):
        for _ in tqdm(range(epoch_nbr)):
            for batch_idx in range(loader.number_of_batches):
                # batch_size x time_steps x features
                batch = loader.load_batch(batch_idx).to_dense()
                batch_in = batch.transpose(0, 1)
                batch_in = batch_in.half()
                self.train_batch(batch_in, loader.effective_steps())

    def encode_docs(self, loader: CUDADataset):
        encoding = []
        self.sdp_learner.disable()
        for batch_idx in tqdm(range(loader.number_of_batches)):
            batch = loader.load_batch(batch_idx).to_dense()
            batch_in = batch.transpose(0, 1)
            encoding.append(self.encode_batch(batch_in, loader.effective_steps()))
        return torch.concat(encoding).numpy()[:loader.document_size]

    def train_batch(self, batch, time_steps, indexes: list = None):
        with torch.no_grad():
            i = 0
            for t in range(time_steps):
                i += 1
                batch_t = batch[t].to(self.device)
                self.optimizer.zero_grad()
                out = self.net(batch_t)
                self.inhibit(out)
                self.sdp_learner.step(on_grad=True)
                self.optimizer.step()
                self.net[0].weight.data.clamp_(self.w_min, self.w_max)
            #   self.net[1].reset_hidden()

    def encode_batch(self, batch, time_steps, inhibition=True):
        with torch.no_grad():
            batch_res = []
            for t in range(time_steps):
                batch_t = batch[t].to(self.device)
                batch_t = batch_t.half()
                spk = self.net(batch_t)
                batch_res.append(spk.cpu())
                if inhibition:
                    self.inhibit(spk)
            return torch.stack(batch_res).sum(0)

    def inhibit(self, out):
        inhibition_mask = out.any(dim=1, keepdims=True).expand_as(out)
        mask = inhibition_mask
        self.net[1].mem[mask] = self.net[1].mem[mask] - self.inhibition

    def encode_docs_dense(self, loader: CUDADataset):
        weights = self.net[0].weight.data.cpu()
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.neuron_nbr),
            snn.Leaky(beta=self.beta,
                      init_hidden=True,
                      inhibition=True,
                      threshold=self.threshold,
                      reset_mechanism='zero')
        )
        self.net[0].weight.data = weights
        print(self.net[0].weight.data.shape)
        self.net[0] = self.net[0].half()
        self.net.to(self.device)
        return self.encode_docs(loader)

    def encode_docs_dense_mini_batch(self, loader: CUDADataset, steps, docs, inhibition=None):
        if inhibition is not None:
            self.inhibition = inhibition
        weights = self.net[0].weight.data.cpu()

        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.neuron_nbr),
            snn.Leaky(beta=0.8,
                      init_hidden=True,
                      inhibition=False,
                      output=False,
                      threshold=self.threshold,
                      reset_mechanism='zero')
        )
        self.net[0].weight.data = weights
        self.net[0] = self.net[0].half()
        self.net.to(self.device)

        small_batch = loader.generate_small_batch(docs).to_dense()

        batch_in = small_batch.transpose(0, 1)
        return self.encode_batch(batch_in, steps, False)

    def doc_cluster_mapping(self, loader: CUDADataset, steps=50):
        topic_map = {idx: [] for idx in range(self.neuron_nbr)}
        print(np.argmax(self.encode_docs(loader, steps), axis=1))
        for idx, top in enumerate(np.argmax(self.encode_docs(loader, steps), axis=1)):
            print(top)
            topic_map[top].append(idx)
        return topic_map

    def topics(self, features, top=10) -> list:

        topics = []
        for i in range(self.neuron_nbr):
            w_after_training = self.net[0].weight.data.cpu().numpy()[i]
            top20 = [(features[idx], w_after_training[idx]) for idx in np.argsort(w_after_training)[-top:]]
            top20 = sorted(top20, key=lambda x: x[1], reverse=True)
            print(top20)
            topics.append([word for word, weight in top20])
        return topics

    def save(self, result_folder: str, name: str):
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        results_path = os.path.join(result_folder, name)
        attributes_dict = self.to_dict(
            ['learning_rate',
             'device',
             'beta',
             'threshold',
             'w_min',
             'w_max',
             'tau_post',
             'squuze',
             'tau_pre',
             'neuron_nbr',
             'input_size'])
        attributes_dict['weights'] = self.net[0].weight.data.cpu().numpy()

        with open(results_path, 'wb') as outp:
            pickle.dump(attributes_dict, outp, pickle.HIGHEST_PROTOCOL)

    def to_dict(self, attribute_names):
        # Create a dictionary from the list of attribute names and their values in self
        return {attr: getattr(self, attr) for attr in attribute_names}

    @staticmethod
    def load(folder, model_name):
        path = os.path.join(folder, model_name)
        network_dict: dict = pickle.load(open(path, "rb"))
        w = network_dict['weights']
        return STMGPUrunner(input_size=network_dict['input_size'],
                            neuron_nbr=network_dict['neuron_nbr'],
                            tau_pre=network_dict['tau_pre'],
                            tau_post=network_dict['tau_post'],
                            beta=network_dict['beta'],
                            squuze=network_dict['squuze'],
                            threshold=network_dict['threshold'],
                            learning_rate=network_dict['learning_rate'],
                            device="cuda",
                            weights_data=w
                            )

    @staticmethod
    def post_norm(x):
        return x * x * 0.2
