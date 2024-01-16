from typing import List, Tuple


class SpikingBatch:

    def __init__(self, stimulus_size: int, features: [] = None):
        self.features = features
        self.stimulus_size = stimulus_size
        self.neuron_indexes: List[int] = []
        self.spiking_times: List[float] = []
        self.simulation_time: float = 0
        self.categories: List[int] = []
        self.input_times: List[Tuple[float, float]] = []
        self.document_times: List[Tuple[float, float]] = []

    def add_input(self, neuron_indexes: [int], spiking_times: [float],
                  start_time: float, end_time: float, category: int):
        self.neuron_indexes.extend(neuron_indexes)
        self.spiking_times.extend(spiking_times)
        self.input_times.append((start_time, end_time))
        #        self.document_times.append((start_time, end_time))
        self.simulation_time = end_time
        self.categories.append(category)

    def add_document_times(self, times: Tuple[float, float]):
        self.document_times.append(times)

    def get_last_time(self):
        if self.input_times:
            return self.input_times[-1][1]
        return 0

    def effective_simulation_time(self):
        return self.simulation_time

    def slice_batch(self, size: int = 10):
        batch: SpikingBatch = SpikingBatch(self.stimulus_size, self.features)
        if len(self.document_times) < size:
            raise Exception("batch size < slice size")
        endTime: int = self.document_times[size - 1][1]
        slice_neurons = []
        slice_times = []
        for (neuron, time) in zip(self.neuron_indexes, self.spiking_times):
            if time > endTime:
                break
            slice_neurons.append(neuron)
            slice_times.append(time)
        batch.spiking_times = slice_times
        batch.neuron_indexes = slice_neurons
        batch.document_times = self.document_times[0:size]
        batch.simulation_time = endTime + 10
        return batch
