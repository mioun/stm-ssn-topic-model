import random

from model.network.spiking_batch import SpikingBatch
from model.sequencing.input_transformer import InputTransformer


class DocumentSequenceTransformer(InputTransformer):

    def __init__(self, size: int,
                 features: list,
                 toke_in_between_time: int = 1,
                 sequence_in_between_time: int = 100,
                 sequence_repeat: int = 10,
                 document_in_between_time: int = 0,
                 minimum_spikes: int = 500,
                 reverse=False):
        self.minimum_spikes = minimum_spikes
        self.reverse = reverse
        self.features = features
        self.toke_in_between_time = toke_in_between_time
        self.document_in_between_time = document_in_between_time
        self.sequence_repeat = sequence_repeat
        self.sequence_in_between_time = sequence_in_between_time
        self.spiking_batch: SpikingBatch = SpikingBatch(size, features)

    def transform(self, documents_dict) -> SpikingBatch:
        print("Preparing spiking batch")
        total_spikes = 0
        avg_doc = 0
        doc_nbr = 0
        for doc, document_sequence in documents_dict.items():
            doc_nbr += 1
            val = [el for el in document_sequence if el[1] > 0]
            if len(val) == 0:
                print(document_sequence)
                continue

            if doc_nbr % 2000 == 0:
                print("Processed : {nbr} , {prctg}% Left : {lefts}".format(nbr=doc_nbr, prctg=round(
                    100 * doc_nbr / len(documents_dict.items()), 2), lefts=len(documents_dict.items()) - doc_nbr))
            avg_doc += len(document_sequence)
            document_start_time = self.spiking_batch.document_times[-1][
                                      1] + 1 if self.spiking_batch.document_times else 0

            neuron_indexes, spike_times = self.doc_to_spikes(document_sequence, document_start_time)
            total_spikes += len(neuron_indexes)
            if len(neuron_indexes) == 0:
                continue
            self.spiking_batch.add_input(neuron_indexes, spike_times, document_start_time,
                                         spike_times[-1] + self.sequence_in_between_time, -1)
            document_end_time = self.spiking_batch.get_last_time() + self.document_in_between_time
            self.spiking_batch.add_document_times((document_start_time, document_end_time))

        return self.spiking_batch

    def doc_to_spikes(self, sentence, sentence_start_time):
        neuron_indexes = []
        current_time = sentence_start_time
        for i in range(self.sequence_repeat):
            neurons = [pair[0] for pair in sentence if pair[1] >= random.random()]
            # if self.reverse and i % 2 == 0:
            #     neurons.reverse()
            neuron_indexes.extend(neurons)
        while len(neuron_indexes) < self.minimum_spikes:
            i = 0
            neurons = [pair[0] for pair in sentence if pair[1] >= random.random()]
            # if self.reverse and i % 2 == 0:
            #     neurons.reverse()
            neuron_indexes.extend(neurons)
            i += 1

        spike_times = [sentence_start_time + i * self.toke_in_between_time for i in range(len(neuron_indexes))]
        return neuron_indexes, spike_times

    def print_sequence(self, seq):
        features_map = {indx: f for indx, f in enumerate(self.features)}
