import os
import pickle
from math import exp
import random
from typing import List

import numpy as np
from brian2 import Network, SpikeGeneratorGroup, NeuronGroup, ms, SpikeMonitor, defaultclock, device, second, set_device
from sklearn.preprocessing import normalize

from model.configuration.configuration_loader import Tribe
from model.network.NeuronGroupWrapper import NeuronGroupWrapper
from model.network.SynapseInitializer import SynapseInitializer
from model.network.spike_generator_group_wrapper import SpikeGeneratorGroupWrapper
from model.network.spiking_batch import SpikingBatch
from model.sequencing.document_sequence_transformer import DocumentSequenceTransformer
from model.topic.topic_object import TopicObject

set_device('cpp_standalone', clean=True)


class STMModelRunner:

    def __init__(self,
                 feature_limit,
                 feature_list: List,
                 freq_map: dict,
                 total_texts: int,
                 encoder_size: int,
                 inh: float,
                 config_path: str,
                 alpha: float,
                 learning_window:
                 int = 50,
                 tau_scaling=100,
                 minimum_spikes=250,
                 eta=0.001):
        self.eta = eta
        self.tau_scaling = tau_scaling
        self.learning_window = learning_window
        self.minimum_spikes = minimum_spikes
        self.alpha = alpha
        self.config_path = config_path
        self.encoder_weights = None
        self.inh = inh
        self.encoder_size = encoder_size
        self.total_texts = total_texts
        self.freq_map = freq_map
        self.feature_list = feature_list
        self.FEATURE_LIMIT = feature_limit

    def extract_topics_from_model(self, word_number=10) -> List[TopicObject]:
        id2token = {idx: feat for idx, feat in enumerate(self.feature_list)}
        topics: List[TopicObject] = []
        weight_matrix = self.weights_to_matrix()
        for id, weights_row in enumerate(weight_matrix):
            word_weight_pairs = [(id2token[idx], val) for idx, val in enumerate(weights_row)]
            word_weight_pairs = sorted(word_weight_pairs, key=lambda x: x[1], reverse=True)
            top_words = [pair[0] for pair in word_weight_pairs[:word_number]]
            topic = TopicObject(id, top_words)
            topic.weights = {pair[0]: pair[1] for pair in word_weight_pairs[:word_number]}
            topics.append(TopicObject(id, top_words))
        return topics

    def compose_brain_network(self, spiking_batch: SpikingBatch, tribe: Tribe, weights_data, inh=None):
        excitatory_yaml = os.path.join(self.config_path, 'encoder_excitatory_layer.yaml')
        excitatory_syn_yaml = os.path.join(self.config_path, 'encoder_excitatory_synapses.yaml')
        inputGroupExt: SpikeGeneratorGroup = SpikeGeneratorGroupWrapper(spiking_batch).build_input()

        excitatory_group: NeuronGroup = NeuronGroupWrapper(self.encoder_size,
                                                           excitatory_yaml,
                                                           tribe,
                                                           'excitatory',
                                                           override_state=self.network_hyper_params(tribe)).group()

        ext_synapses = SynapseInitializer(excitatory_syn_yaml, len(self.feature_list), self.encoder_size,
                                          inputGroupExt, excitatory_group, weights_data=weights_data, w_max=1,
                                          tribe=tribe, name="encoder",
                                          override_state=self.synapse_hyper_params(tribe)).synapses()

        network: Network = Network(inputGroupExt, excitatory_group, ext_synapses)
        effective_inh = inh if inh is not None else self.inh
        if effective_inh != 0:
            excitatory_excitatory_synapses_yaml = os.path.join(self.config_path,
                                                               'encoder_inhibitory_synapses.yaml')
            lateral_inhibition_synapses = SynapseInitializer(excitatory_excitatory_synapses_yaml, self.encoder_size,
                                                             self.encoder_size, excitatory_group, excitatory_group,
                                                             w_max=1,
                                                             weight_val=effective_inh,
                                                             tribe=tribe, name="ext_inh", all_not_me=True).synapses()
            network.add(lateral_inhibition_synapses)
        if tribe is Tribe.TEST:
            spikeMon: SpikeMonitor = SpikeMonitor(excitatory_group, name='spike_monitor')
            network.add(spikeMon)
        return network

    def synapse_hyper_params(self, tribe) -> dict:
        if Tribe.TRAIN == tribe:
            return {'eta': self.eta, 'taupre': self.learning_window * ms}
        else:
            return {}

    def network_hyper_params(self, tribe) -> dict:
        if Tribe.TRAIN == tribe:
            return {'tau_scaling': self.tau_scaling * ms}
        else:
            return {}

    def train_epoch(self, doc_tokens: List):
        input_spiking_batch: SpikingBatch = self.prepare_batch(input_size=self.FEATURE_LIMIT,
                                                               input_features=self.feature_list,
                                                               freq_map=self.freq_map,
                                                               training_seqs=doc_tokens,
                                                               minimum_spikes=self.minimum_spikes,
                                                               total_doc_number=self.total_texts)
        network: Network = self.compose_brain_network(input_spiking_batch, Tribe.TRAIN, self.encoder_weights)
        defaultclock.dt = 1 * ms
        network.run(input_spiking_batch.effective_simulation_time() * ms, report='text', report_period=60 * second)
        self.encoder_weights = self.get_weights_data(network, 'encoder')
        device.reinit()
        device.activate()

    def train(self, epoch_nbr: int, doc_tokens):
        for e in range(epoch_nbr):
            print(f'#### Training epoch {e + 1} of {epoch_nbr}')
            docs = doc_tokens.copy()
            random.shuffle(docs)
            self.train_epoch(docs)

    def represent(self, doc_tokens: List, inhibition=None):
        input_spiking_batch: SpikingBatch = self.prepare_batch(input_size=self.FEATURE_LIMIT,
                                                               input_features=self.feature_list,
                                                               freq_map=self.freq_map,
                                                               training_seqs=doc_tokens,
                                                               minimum_spikes=self.minimum_spikes,
                                                               total_doc_number=self.total_texts)
        network: Network = self.compose_brain_network(input_spiking_batch, Tribe.TEST, self.encoder_weights,
                                                      inh=inhibition)
        defaultclock.dt = 1 * ms
        network.run(input_spiking_batch.effective_simulation_time() * ms, report='text', report_period=60 * second)
        monitor: SpikeMonitor = network.get_states()['spike_monitor']
        times, neurons = monitor['t'] / ms, monitor['i']
        device.reinit()
        device.activate()
        return self.represent_spikes_as_matrix(input_spiking_batch, times, neurons, self.encoder_size)

    def represent_norm(self, doc_tokens: List, inhibition=None):
        return normalize(self.represent(doc_tokens, inhibition))

    def prepare_batch(self, input_size: int, input_features: list, freq_map: dict, training_seqs: list,
                      total_doc_number: int, minimum_spikes) -> SpikingBatch:
        documents_dict = self.to_token_freq_dict(training_seqs, input_features, freq_map, total_doc_number)
        transformer: DocumentSequenceTransformer = DocumentSequenceTransformer(input_size,
                                                                               input_features,
                                                                               sequence_repeat=1,
                                                                               reverse=True,
                                                                               minimum_spikes=minimum_spikes,
                                                                               document_in_between_time=1000)
        return transformer.transform(documents_dict)

    def get_weights_data(self, network: Network, name: str) -> dict:
        if name not in network.get_states():
            return None
        weights = network.get_states()[name]['w']
        i = network.get_states()[name]['i']
        j = network.get_states()[name]['j']
        return {
            'weights': weights,
            'i': i,
            'j': j}

    def calculate_spike_probability(self, feature_freq, doc_nbr, threshold=0.000):
        freq_threshold = threshold * doc_nbr
        return exp(self.alpha * feature_freq)

    def to_token_freq_dict(self, tokenize_docs: list, features_list: list, frequency_map: dict,
                           total_doc_nbr: int) -> dict:
        documents_dict = {}
        neuron_word_map = {word: idx for idx, word in enumerate(features_list)}
        for doc_nbr, tokens in enumerate(tokenize_docs):
            document_representation = []
            for token in tokens:
                spike_prob = self.calculate_spike_probability(frequency_map[token], total_doc_nbr)
                document_representation.append((neuron_word_map[token], spike_prob))
            documents_dict[doc_nbr] = document_representation
        return documents_dict

    def weights_to_matrix(self):
        weights_matrix = np.zeros((max(self.encoder_weights['j']) + 1, max(self.encoder_weights['i'] + 1)))
        for idx, val in enumerate(self.encoder_weights['j']):
            weights_matrix[val][self.encoder_weights['i'][idx]] = self.encoder_weights['weights'][idx]
        return weights_matrix

    def save(self, result_folder, name=''):
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        results_path = os.path.join(result_folder, name)
        with open(results_path, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def represent_spikes_as_matrix(self, spiking_batch: SpikingBatch, spike_times: list = None, neurons: list = None,
                                   custom_neuron_number=None):
        if spike_times is None:
            spike_times = spiking_batch.spiking_times
        if neurons is None:
            neurons = spiking_batch.neuron_indexes

        if len(spiking_batch.document_times) > 0:
            documents_end_times = [t[1] for t in spiking_batch.document_times]
        else:
            documents_end_times = [t[1] for t in spiking_batch.input_times]
        doc_count = len(documents_end_times)
        output_neuron_number = spiking_batch.stimulus_size if custom_neuron_number is None else custom_neuron_number
        document_representation = np.zeros((doc_count, output_neuron_number))
        current_doc = 0
        for idx, time in enumerate(spike_times):
            neuron_idx = neurons[idx]
            while time > documents_end_times[current_doc]:
                current_doc += 1
            document_representation[current_doc][neuron_idx] += 1
        return document_representation

    @staticmethod
    def load(folder, model_name):
        path = os.path.join(folder, model_name)
        modelObject = pickle.load(open(path, "rb"))
        return modelObject
