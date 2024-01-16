import pickle
from typing import Any

from brian2 import Synapses, SpikeSource, np

from model.configuration.configuration_loader import ConfigurationLoader, EquationCategory, Tribe


class SynapseInitializer:

    def __init__(self, config_file: str, pre_synaptic_size: int, post_synaptic_size: int,
                 source: SpikeSource, target: Any, tribe: Tribe = Tribe.TRAIN, w_max=0.2, name=None,
                 weight_val=None, conn_probability: float = None, weights_data=None, all_not_me=False,
                 override_state=None):

        self.override_state = {} if override_state is None else override_state
        self.all_not_me = all_not_me
        self.weights_data = weights_data
        self.postsynaptic_size = post_synaptic_size
        self.presynaptic_size = pre_synaptic_size
        self.conn_probability = conn_probability
        self.weight_val = weight_val
        self.target: Any = target
        self.conf_loader: ConfigurationLoader = ConfigurationLoader(config_file)
        self.tribe = tribe
        self.source: SpikeSource = source
        self.w_max = w_max
        model_equations = self.conf_loader.load_equations_if_preset(self.tribe, EquationCategory.MODEL)
        on_pre_equations = self.conf_loader.load_equations_if_preset(self.tribe, EquationCategory.PRESYNAPTIC_EVENT)
        on_post_equations = self.conf_loader.load_equations_if_preset(self.tribe, EquationCategory.POSTSYNAPTIC_EVENT)
        if name:
            self.syn: Synapses = Synapses(self.source, self.target, model=model_equations, on_pre=on_pre_equations,
                                          name=name,
                                          on_post=on_post_equations)
        else:
            self.syn: Synapses = Synapses(self.source, self.target, model=model_equations, on_pre=on_pre_equations,
                                          on_post=on_post_equations)
        self.connection_strategy()
        self.set_states()

    def connection_strategy(self):
        if self.weights_data:
            self.syn.connect(i=self.weights_data['i'], j=self.weights_data['j'])
        elif self.conn_probability:
            self.syn.connect(p=self.conn_probability)
        elif self.all_not_me:
            self.syn.connect(condition='i != j')
        else:
            i, j = self.generate_connections()
            self.syn.connect(i=i, j=j)

    def synapses(self) -> Synapses:
        return self.syn

    def set_states(self):
        state = self.conf_loader.load_state_if_exist(self.tribe)
        for key in self.override_state:
            state[key] = self.override_state[key]
        if state:
            self.syn.set_states(state)

        if self.weights_data is not None:
            self.syn.w = self.weights_data['weights']
        elif self.weight_val:
            self.syn.w = self.weight_val
        else:
            self.syn.w = self.generate_random_weights().flatten()

    def generate_random_weights(self) -> []:
        return np.random.rand(self.presynaptic_size, self.postsynaptic_size) * self.w_max

    def generate_connections(self) -> [[int], [int]]:
        i = [i % self.presynaptic_size for i in range(0, self.presynaptic_size * self.postsynaptic_size)]
        j = []
        current_post_idx = -1
        for syn_idx in range(0, self.presynaptic_size * self.postsynaptic_size):
            if syn_idx % self.presynaptic_size == 0:
                current_post_idx += 1
            j.append(current_post_idx)
        return i, j


def extract_synaptic_data(syn):
    weights = syn.get_states()['w']
    train_batch = {}
    train_batch['weights'] = weights
    train_batch['i'] = syn.get_states()['i']
    train_batch['j'] = syn.get_states()['j']
    return train_batch
