from model.network.spiking_batch import SpikingBatch


class InputTransformer:

    dt: float = 1.0

    def transform(self, input_array) -> SpikingBatch:
        pass
