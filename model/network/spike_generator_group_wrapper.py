from brian2 import SpikeGeneratorGroup, ms

from model.network.spiking_batch import SpikingBatch


class SpikeGeneratorGroupWrapper:

    def __init__(self, spiking_batch: SpikingBatch):
        self.spiking_batch: SpikingBatch = spiking_batch

    def build_input(self) -> SpikeGeneratorGroup:
        print(len(self.spiking_batch.input_times), len(self.spiking_batch.neuron_indexes))
        group: SpikeGeneratorGroup = SpikeGeneratorGroup(
            N=self.spiking_batch.stimulus_size,
            times=self.spiking_batch.spiking_times * ms,
            indices=self.spiking_batch.neuron_indexes)
        return group
