import random

from model.network.spiking_batch import SpikingBatch
from model.sequencing.input_transformer import InputTransformer


class DocumentSequenceTransformerW2Vec(InputTransformer):

    def __init__(self, size: int,
                 features: list,
                 toke_in_between_time: int = 1,
                 sequence_in_between_time: int = 100,
                 sequence_repeat: int = 10,
                 document_in_between_time: int = 0,
                 minimum_spikes: int = 500,
                 probability_dict=None,
                 reverse=False):
        self.probability_dict = probability_dict
        self.minimum_spikes = minimum_spikes
        self.reverse = reverse
        self.features = features
        self.toke_in_between_time = toke_in_between_time
        self.document_in_between_time = document_in_between_time
        self.sequence_repeat = sequence_repeat
        self.sequence_in_between_time = sequence_in_between_time
        self.spiking_batch: SpikingBatch = SpikingBatch(size, features)
        self.probability_table = [0] * len(features)

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
            input("Presss enter")
        return self.spiking_batch

    def doc_to_spikes(self, sentence, sentence_start_time):
        neuron_indexes = []
        current_time = sentence_start_time
        print("####################Start doc")
        for i in range(self.sequence_repeat):
            neurons = self.generate_indxs_per_word(sentence)
            if self.reverse and i % 2 == 0:
                neurons.reverse()
            neuron_indexes.extend(neurons)
        while len(neuron_indexes) < self.minimum_spikes:
            i = 0
            neurons = self.generate_indxs_per_word(sentence)
            if self.reverse and i % 2 == 0:
                neurons.reverse()
            neuron_indexes.extend(neurons)
            i += 1
        print("########################END doc")
        spike_times = [sentence_start_time + i * self.toke_in_between_time for i in range(len(neuron_indexes))]
        return neuron_indexes, spike_times

    def generate_indxs_per_word(self, sentence):
        sentence_idxs = []
        for pair in sentence:
            feature = self.features[pair[0]]
            feature_probs = self.probability_dict[feature]
            if pair[1] >= random.random():
                sentence_idxs.append(pair[0])
                print(self.features[pair[0]])
                for ctx in feature_probs:
                    if ctx[0] == pair[0]:
                        continue
                    self.probability_table[ctx[0]] += ctx[1]*0.2
                max = 0
                id = 0
                for idx, val in enumerate(self.probability_table):
                    if val > max:
                        max = val
                        id = idx

                if max > 2:
                    print("Extra spike",self.features[pair[0]], self.features[id])
                    self.probability_table[id] = 0
                    self.probability_table = [x * 0.9 for x in self.probability_table]
                    sentence_idxs.append(id)
            self.probability_table = [x * 0.9 for x in self.probability_table]
                # context_idcs = [trio[1] for trio in feature_probs[1:] if trio[2] * 0.1 >= random.random()]
                # sentence_idxs.extend(context_idcs)
                # if len(context_idcs) > 0:
                #     print("There is background : ", self.features[pair[0]],
                #           [self.features[idx] for idx in context_idcs])
        return sentence_idxs

    def print_sequence(self, seq):
        features_map = {indx: f for indx, f in enumerate(self.features)}
