from brian2 import NeuronGroup


from model.configuration.configuration_loader import EquationCategory, ConfigurationLoader, Tribe


class NeuronGroupWrapper:

    def __init__(self, network_size: int, configuration_file: str, tribe: Tribe, name=None,
                 override_state=None):
        if override_state is None:
            override_state = {}
        self.override_state = override_state
        self.name = name
        self.tribe = tribe
        self.network_size = network_size
        self.configurationFile = configuration_file
        self.conf_loader: ConfigurationLoader = ConfigurationLoader(configuration_file)
        self.neuronGroup: NeuronGroup = None

    def group(self) -> NeuronGroup:
        if self.name:
            self.neuronGroup: NeuronGroup = NeuronGroup(
                N=self.network_size,
                model=self.conf_loader.load_equations(self.tribe, EquationCategory.MODEL),
                reset=self.conf_loader.load_equations(self.tribe, EquationCategory.RESET),
                refractory=self.conf_loader.load_equations(self.tribe, EquationCategory.REFRACTORY),
                method=self.conf_loader.load_equations(self.tribe, EquationCategory.METHOD),
                threshold=self.conf_loader.load_equations(self.tribe, EquationCategory.TRESH_HOLD),
                name=self.name)
        else:
            self.neuronGroup: NeuronGroup = NeuronGroup(
                N=self.network_size,
                model=self.conf_loader.load_equations(self.tribe, EquationCategory.MODEL),
                reset=self.conf_loader.load_equations(self.tribe, EquationCategory.RESET),
                refractory=self.conf_loader.load_equations(self.tribe, EquationCategory.REFRACTORY),
                method=self.conf_loader.load_equations(self.tribe, EquationCategory.METHOD),
                threshold=self.conf_loader.load_equations(self.tribe, EquationCategory.TRESH_HOLD))
        state = self.conf_loader.load_state(self.tribe)

        for i, (key, val) in enumerate(self.override_state.items()):
            state[key] = val
        self.neuronGroup.set_states(state)
        return self.neuronGroup
