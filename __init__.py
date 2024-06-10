from .sigma_waveform_node import SigmaWaveFormNodeSimple, SigmaWaveFormNode, SigmaWaveFormNodeAdvanced, FourierFilterNode, PhaseLockedLoopNode, AttenuatorNode


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "SigmaWaveFormNodeSimple": SigmaWaveFormNodeSimple,
    "SigmaWaveFormNode": SigmaWaveFormNode,
    "SigmaWaveFormNodeAdvanced": SigmaWaveFormNodeAdvanced,
    "FourierFilterNode": FourierFilterNode,
    "PhaseLockedLoopNode": PhaseLockedLoopNode, 
    "AttenuatorNode": AttenuatorNode
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "SigmaWaveFormNodeSimple": "Sigma Waveform Node Simple",
    "SigmaWaveFormNode": "Sigma Waveform Node",
    "SigmaWaveFormNodeAdvanced": "Sigma Waveform Node Advanced",
    "FourierFilterNode": "Fourier Filter Node",
    "PhaseLockedLoopNode": "Phase Locked Loop Node",
    "AttenuatorNode": "Attenuator Node"
}
