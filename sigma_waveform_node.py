import numpy as np
import torch


def loglinear_interp(t_steps, num_steps):
    """
    Performs log-linear interpolation of a given array of decreasing numbers.
    """
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])

    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys

def RESDECoeffsSecondOrder(h):
    """
    Calculate the coefficients for the RES solver.
    """
    b1 = 1.0 - 0.5 * h
    b2 = 0.5 * h
    return b1, b2

class SigmaWaveFormNodeSimple:
    def __init__(self):
        self.steps = 200
        self.amplitude = 3.0
        self.frequency = 0.1
        self.damping_factor = 0.03
        self.waveform_type = "sine"
        self.enable_unidirectional = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 200, "min": 1, "max": 1000, "step": 1}),
                "amplitude": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "frequency": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "damping_factor": ("FLOAT", {"default": 0.03, "min": 0.001, "max": 0.1, "step": 0.001}),
                "waveform_type": (["sine", "sawtooth", "stepped", "square", "random"], {"default": "sine"}),
                "enable_unidirectional": ("BOOLEAN", {"default": True, "description": "Whether to make the waveform unidirectional"}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "generate_sigma_sequence"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def generate_sine_wave(self, t):
        return self.amplitude * np.exp(-self.damping_factor * t) * np.sin(2 * np.pi * self.frequency * t)
    
    def generate_sawtooth_wave(self, t):
        return self.amplitude * np.exp(-self.damping_factor * t) * 2 * (t * self.frequency - np.floor(0.5 + t * self.frequency))
    
    def generate_stepped_wave(self, t):
        steps_per_segment = self.steps // 10
        return np.repeat(np.linspace(self.amplitude, -self.amplitude, 10), steps_per_segment)
    
    def generate_square_wave(self, t):
        return self.amplitude * np.exp(-self.damping_factor * t) * np.sign(np.sin(2 * np.pi * self.frequency * t))
    
    def generate_random_wave(self, t):
        return self.amplitude * np.exp(-self.damping_factor * t) * (2 * np.random.rand(self.steps) - 1)

    def make_unidirectional(self, sigma):
        return sigma - np.min(sigma)

    def generate_sigma_sequence(self, steps, amplitude, frequency, damping_factor, waveform_type, enable_unidirectional):
        self.steps = steps
        self.amplitude = amplitude
        self.frequency = frequency
        self.damping_factor = damping_factor
        self.waveform_type = waveform_type
        self.enable_unidirectional = enable_unidirectional

        t = np.linspace(0, steps, steps)
        if waveform_type == "sine":
            sigma = self.generate_sine_wave(t)
        elif waveform_type == "sawtooth":
            sigma = self.generate_sawtooth_wave(t)
        elif waveform_type == "stepped":
            sigma = self.generate_stepped_wave(t)
        elif waveform_type == "square":
            sigma = self.generate_square_wave(t)
        elif waveform_type == "random":
            sigma = self.generate_random_wave(t)
        else:
            sigma = np.zeros(steps)  # Default case

        if enable_unidirectional:
            sigma = self.make_unidirectional(sigma)

        # Ensure sigma values are in the correct format
        sigma_tensor = torch.tensor(sigma, dtype=torch.float64)
        return (sigma_tensor,)

    def update_parameters(self, steps=None, amplitude=None, frequency=None, damping_factor=None, waveform_type=None, enable_unidirectional=None):
        if steps is not None:
            self.steps = steps
        if amplitude is not None:
            self.amplitude = amplitude
        if frequency is not None:
            self.frequency = frequency
        if damping_factor is not None:
            self.damping_factor = damping_factor
        if waveform_type is not None:
            self.waveform_type = waveform_type
        if enable_unidirectional is not None:
            self.enable_unidirectional = enable_unidirectional

        return self.generate_sigma_sequence(
            self.steps, 
            self.amplitude, 
            self.frequency, 
            self.damping_factor, 
            self.waveform_type,
            self.enable_unidirectional
        )
            

class SigmaWaveFormNode:
    def __init__(self):
        self.steps = 60
        self.amplitude = 3.0
        self.frequency = 0.1
        self.damping_factor = 0.03
        self.waveform_type = "sine"
        self.enable_damping = True
        self.min_sigma = 0.0
        self.apply_fourier = False
        self.horizontal_length = 0.1
        self.enable_unidirectional = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 60, "min": 1, "max": 1000, "step": 1}),
                "amplitude": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 100.0, "step": 0.1, "description": "Amplitude of the waveform"}),
                "frequency": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 10.0, "step": 0.01, "description": "Frequency of the waveform"}),
                "waveform_type": (["sine", "sawtooth", "stepped", "square", "random", "triangle", "rectangular"], {"default": "sine", "description": "Type of the waveform"}),
                "damping_factor": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.1, "step": 0.001, "description": "Damping factor for the waveform"}),
                "enable_damping": ("BOOLEAN", {"default": True, "description": "Whether to apply damping to the waveform"}),
                "min_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1, "description": "Minimum sigma value"}),
                "apply_fourier": ("BOOLEAN", {"default": False, "description": "Whether to apply Fourier transform to the waveform"}),
                "horizontal_length": ("FLOAT", {"default": 0.1, "min": 0.00, "max": 1.0, "step": 0.01, "description": "Proportion of each horizontal segment length in the stepped waveform"}),
                "enable_unidirectional": ("BOOLEAN", {"default": True, "description": "Whether to make the waveform unidirectional"}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "generate_sigma_sequence"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def apply_damping(self, t, values):
        if self.enable_damping:
            return values * np.exp(-self.damping_factor * t)
        return values

    def generate_waveform(self, waveform_type, amplitude, frequency, t):
        if waveform_type == "sine":
            values = amplitude * np.sin(2 * np.pi * frequency * t)
        elif waveform_type == "sawtooth":
            values = amplitude * 2 * (t * frequency - np.floor(0.5 + t * frequency))
        elif waveform_type == "stepped":
            num_segments = int(1 / self.horizontal_length)
            steps_per_segment = self.steps // num_segments
            values = np.repeat(np.linspace(amplitude, -amplitude, num_segments), steps_per_segment)
            values = np.resize(values, self.steps)
        elif waveform_type == "square":
            values = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
        elif waveform_type == "random":
            values = amplitude * (2 * np.random.rand(self.steps) - 1)
        elif waveform_type == "triangle":
            values = amplitude * (2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1)
        elif waveform_type == "rectangular":
            values = amplitude * (np.mod(t * frequency, 1) < 0.5).astype(float)
        else:
            values = np.zeros(self.steps)
        return values

    def apply_fourier_transform(self, values):
        fft_vals = np.fft.fft(values)
        fft_vals[5:] = 0
        values = np.fft.ifft(fft_vals).real
        return values

    def make_unidirectional(self, values):
        return values - np.min(values) + self.min_sigma

    def generate_sigma_sequence(self, steps, amplitude, frequency, waveform_type, damping_factor, enable_damping, min_sigma, apply_fourier, horizontal_length, enable_unidirectional):
        self.steps = steps
        self.amplitude = amplitude
        self.frequency = frequency
        self.waveform_type = waveform_type
        self.damping_factor = damping_factor
        self.enable_damping = enable_damping
        self.min_sigma = min_sigma
        self.apply_fourier = apply_fourier
        self.horizontal_length = horizontal_length
        self.enable_unidirectional = enable_unidirectional

        t = np.linspace(0, 1, steps)

        sigma = self.generate_waveform(waveform_type, amplitude, frequency, t)

        if enable_unidirectional:
            sigma = self.make_unidirectional(sigma)

        if enable_damping:
            sigma = self.apply_damping(t, sigma)

        if apply_fourier:
            sigma = self.apply_fourier_transform(sigma)

        sigma_tensor = torch.tensor(sigma, dtype=torch.float64)
        return (sigma_tensor,)

    def update_parameters(self, steps=None, amplitude=None, frequency=None, waveform_type=None, damping_factor=None, enable_damping=None, min_sigma=None, apply_fourier=None, horizontal_length=None, enable_unidirectional=None):
        if steps is not None:
            self.steps = steps
        if amplitude is not None:
            self.amplitude = amplitude
        if frequency is not None:
            self.frequency = frequency
        if waveform_type is not None:
            self.waveform_type = waveform_type
        if damping_factor is not None:
            self.damping_factor = damping_factor
        if enable_damping is not None:
            self.enable_damping = enable_damping
        if min_sigma is not None:
            self.min_sigma = min_sigma
        if apply_fourier is not None:
            self.apply_fourier = apply_fourier
        if horizontal_length is not None:
            self.horizontal_length = horizontal_length
        if enable_unidirectional is not None:
            self.enable_unidirectional = enable_unidirectional

        return self.generate_sigma_sequence(
            self.steps, 
            self.amplitude, 
            self.frequency, 
            self.waveform_type, 
            self.damping_factor, 
            self.enable_damping, 
            self.min_sigma, 
            self.apply_fourier, 
            self.horizontal_length,
            self.enable_unidirectional
        )


class SigmaWaveFormNodeAdvanced:
    def __init__(self):
        self.steps = 60
        self.amplitude1 = 3.0
        self.frequency1 = 0.1
        self.amplitude2 = 3.0
        self.frequency2 = 0.1
        self.damping_factor = 0.03
        self.waveform_type1 = "sine"
        self.waveform_type2 = "sine"
        self.blend_factor = 0.5
        self.enable_damping = True
        self.min_sigma = 0.0
        self.apply_fourier = False
        self.horizontal_length = 0.1
        self.apply_loglinear = False
        self.noise_fine_tune = 1.0
        self.enable_unidirectional = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 60, "min": 1, "max": 1000, "step": 1}),
                "noise_fine_tune": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "description": "Fine-tuning factor for noise levels"}),
                "waveform_type1": (["sine", "sawtooth", "stepped", "square", "random", "triangle", "rectangular"], {"default": "sine", "description": "Type of the first waveform"}),
                "waveform_type2": (["sine", "sawtooth", "stepped", "square", "random", "triangle", "rectangular"], {"default": "sine", "description": "Type of the second waveform"}),
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "description": "Blending factor between the two waveforms"}),
                "amplitude1": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 100.0, "step": 0.1, "description": "Amplitude of the first waveform"}),
                "amplitude2": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 100.0, "step": 0.1, "description": "Amplitude of the second waveform"}),
                "frequency1": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 10.0, "step": 0.01, "description": "Frequency of the first waveform"}),
                "frequency2": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 10.0, "step": 0.01, "description": "Frequency of the second waveform"}),
                "horizontal_length": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01, "description": "Proportion of each horizontal segment length in the stepped waveform"}),
                "min_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1, "description": "Minimum sigma value"}),
                "apply_fourier": ("BOOLEAN", {"default": False, "description": "Whether to apply Fourier transform to the waveform"}),
                "apply_loglinear": ("BOOLEAN", {"default": False, "description": "Whether to apply log-linear interpolation to the sigma values"}),
                "enable_damping": ("BOOLEAN", {"default": True, "description": "Whether to apply damping to the waveform"}),
                "damping_factor": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.1, "step": 0.001, "description": "Damping factor for the waveform"}),
                "enable_unidirectional": ("BOOLEAN", {"default": True, "description": "Whether to make the waveform unidirectional"})  
                
                
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "generate_sigma_sequence"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def apply_damping(self, t, values):
        if self.enable_damping:
            return values * np.exp(-self.damping_factor * t)
        return values

    def generate_waveform(self, waveform_type, amplitude, frequency, t):
        if waveform_type == "sine":
            values = amplitude * np.sin(2 * np.pi * frequency * t)
        elif waveform_type == "sawtooth":
            values = amplitude * 2 * (t * frequency - np.floor(0.5 + t * frequency))
        elif waveform_type == "stepped":
            num_segments = int(1 / self.horizontal_length)
            steps_per_segment = self.steps // num_segments
            values = np.repeat(np.linspace(amplitude, -amplitude, num_segments), steps_per_segment)
            values = np.resize(values, self.steps)
        elif waveform_type == "square":
            values = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
        elif waveform_type == "random":
            values = amplitude * (2 * np.random.rand(self.steps) - 1)
        elif waveform_type == "triangle":
            values = amplitude * (2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1)
        elif waveform_type == "rectangular":
            values = amplitude * (np.mod(t * frequency, 1) < 0.5).astype(float)
        else:
            values = np.zeros(self.steps)
        return values

    def apply_fourier_transform(self, values):
        fft_vals = np.fft.fft(values)
        fft_vals[5:] = 0
        values = np.fft.ifft(fft_vals).real
        return values

    def make_unidirectional(self, values):
        return values - np.min(values) + self.min_sigma

    def refined_exponential_solver(self, sigma, steps):
        """
        Apply the refined exponential solver for the sigma sequence.
        """
        h = 1 / steps
        b1, b2 = RESDECoeffsSecondOrder(h)
        x = np.zeros(steps)
        for n in range(steps - 1):
            x[n + 1] = x[n] + h * (b1 * sigma[n] + b2 * sigma[n + 1])
        return x

    def generate_sigma_sequence(self, steps, amplitude1, frequency1, waveform_type1, amplitude2, frequency2, waveform_type2, blend_factor, damping_factor, enable_damping, min_sigma, apply_fourier, horizontal_length, apply_loglinear, noise_fine_tune, enable_unidirectional):
        self.steps = steps
        self.amplitude1 = amplitude1
        self.frequency1 = frequency1
        self.waveform_type1 = waveform_type1
        self.amplitude2 = amplitude2
        self.frequency2 = frequency2
        self.waveform_type2 = waveform_type2
        self.blend_factor = blend_factor
        self.damping_factor = damping_factor
        self.enable_damping = enable_damping
        self.min_sigma = min_sigma
        self.apply_fourier = apply_fourier
        self.horizontal_length = horizontal_length
        self.apply_loglinear = apply_loglinear
        self.noise_fine_tune = noise_fine_tune
        self.enable_unidirectional = enable_unidirectional

        t = np.linspace(0, 1, steps)

        sigma1 = self.generate_waveform(waveform_type1, amplitude1, frequency1, t)
        sigma2 = self.generate_waveform(waveform_type2, amplitude2, frequency2, t)
        sigma = blend_factor * sigma1 + (1 - blend_factor) * sigma2

        if self.enable_unidirectional:
            sigma = self.make_unidirectional(sigma)

        if enable_damping:
            sigma = self.apply_damping(t, sigma)

        if apply_fourier:
            sigma = self.apply_fourier_transform(sigma)

        if apply_loglinear:
            sigma = loglinear_interp(sigma, steps)

        sigma *= noise_fine_tune

        sigma = self.refined_exponential_solver(sigma, steps)

        sigma_tensor = torch.tensor(sigma, dtype=torch.float64)
        return (sigma_tensor,)

    def update_parameters(self, amplitude1=None, frequency1=None, amplitude2=None, frequency2=None, blend_factor=None, damping_factor=None, horizontal_length=None, apply_loglinear=None, noise_fine_tune=None, enable_unidirectional=None):
        if amplitude1 is not None:
            self.amplitude1 = amplitude1
        if frequency1 is not None:
            self.frequency1 = frequency1
        if amplitude2 is not None:
            self.amplitude2 = amplitude2
        if frequency2 is not None:
            self.frequency2 = frequency2
        if blend_factor is not None:
            self.blend_factor = blend_factor
        if damping_factor is not None:
            self.damping_factor = damping_factor
        if horizontal_length is not None:
            self.horizontal_length = horizontal_length
        if apply_loglinear is not None:
            self.apply_loglinear = apply_loglinear
        if noise_fine_tune is not None:
            self.noise_fine_tune = noise_fine_tune
        if enable_unidirectional is not None:
            self.enable_unidirectional = enable_unidirectional   

        self.generate_sigma_sequence(
            self.steps, 
            self.amplitude1, 
            self.frequency1, 
            self.waveform_type1, 
            self.amplitude2, 
            self.frequency2, 
            self.waveform_type2, 
            self.blend_factor, 
            self.damping_factor, 
            self.enable_damping, 
            self.min_sigma, 
            self.apply_fourier, 
            self.horizontal_length,
            self.apply_loglinear,
            self.noise_fine_tune,
            self.enable_unidirectional
        )

import numpy as np
import torch

class FourierFilterNode:
    def __init__(self):
        self.filter_type = "lowpass"
        self.cutoff_frequency = 0.1
        self.apply_inverse_fourier = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"description": "Input sigma sequence"}),
                "filter_type": (["lowpass", "highpass", "bandpass", "bandstop"], {"default": "lowpass", "description": "Type of frequency filter to apply"}),
                "cutoff_frequency": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01, "description": "Cutoff frequency for the filter"}),
                "apply_inverse_fourier": ("BOOLEAN", {"default": True, "description": "Whether to apply inverse Fourier transform"}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("filtered_sigmas",)
    FUNCTION = "apply_filter"
    CATEGORY = "sampling/custom_sampling/filters"

    def apply_filter(self, sigmas, filter_type, cutoff_frequency, apply_inverse_fourier):
        self.filter_type = filter_type
        self.cutoff_frequency = cutoff_frequency
        self.apply_inverse_fourier = apply_inverse_fourier

        # Convert input tensor to numpy array
        sigma = sigmas.numpy()

        # Apply Fourier Transform
        fft_vals = np.fft.fft(sigma)

        # Apply the selected filter
        filtered_fft_vals = self.apply_frequency_filter(fft_vals, filter_type, cutoff_frequency)

        # Optionally apply inverse Fourier transform
        if apply_inverse_fourier:
            filtered_sigma = np.fft.ifft(filtered_fft_vals).real
        else:
            filtered_sigma = filtered_fft_vals

        # Convert filtered result back to tensor
        filtered_sigma_tensor = torch.tensor(filtered_sigma, dtype=torch.float32)
        return (filtered_sigma_tensor,)

    def apply_frequency_filter(self, fft_vals, filter_type, cutoff_frequency):
        freqs = np.fft.fftfreq(len(fft_vals))
        filtered_fft_vals = np.copy(fft_vals)

        if filter_type == "lowpass":
            filtered_fft_vals[np.abs(freqs) > cutoff_frequency] = 0
        elif filter_type == "highpass":
            filtered_fft_vals[np.abs(freqs) < cutoff_frequency] = 0
        elif filter_type == "bandpass":
            band = (cutoff_frequency, cutoff_frequency * 2)  # Example band range, can be adjusted
            filtered_fft_vals[(np.abs(freqs) < band[0]) | (np.abs(freqs) > band[1])] = 0
        elif filter_type == "bandstop":
            band = (cutoff_frequency, cutoff_frequency * 2)  # Example band range, can be adjusted
            filtered_fft_vals[(np.abs(freqs) > band[0]) & (np.abs(freqs) < band[1])] = 0

        return filtered_fft_vals



class AttenuatorNode:
    def __init__(self):
        self.attenuation_factor = 1.0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"description": "Input sigma sequence"}),
                "attenuation_factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1, "description": "Factor by which to attenuate the sigma sequence"}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("attenuated_sigmas",)
    FUNCTION = "apply_attenuation"
    CATEGORY = "sampling/custom_sampling/attenuators"

    def apply_attenuation(self, sigmas, attenuation_factor):
        self.attenuation_factor = attenuation_factor

        # Apply attenuation
        attenuated_sigma = sigmas * attenuation_factor

        return (attenuated_sigma,)


class PhaseLockedLoopNode:
    def __init__(self):
        self.lock_factor = 0.5
        self.frequency_range = 1.0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_sigmas": ("SIGMAS", {"description": "Input sigma sequence"}),
                "reference_sigmas": ("SIGMAS", {"description": "Reference sigma sequence"}),
                "lock_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "description": "Factor by which to lock the phase"}),
                "frequency_range": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "description": "Frequency range for the PLL"}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("locked_sigmas",)
    FUNCTION = "lock_phase"
    CATEGORY = "sampling/custom_sampling/phase_locked_loop"

    def lock_phase(self, input_sigmas, reference_sigmas, lock_factor, frequency_range):
        self.lock_factor = lock_factor
        self.frequency_range = frequency_range

        # Convert tensors to numpy arrays
        input_sigma = input_sigmas.numpy()
        reference_sigma = reference_sigmas.numpy()

        # Initialize output signal
        locked_sigma = np.zeros_like(input_sigma)

        # Phase Locked Loop logic (simplified example)
        phase_error = 0
        for i in range(len(input_sigma)):
            phase_error += reference_sigma[i] - input_sigma[i]
            correction = self.lock_factor * phase_error
            locked_sigma[i] = input_sigma[i] + correction

        # Apply frequency range limits
        locked_sigma = np.clip(locked_sigma, -self.frequency_range, self.frequency_range)

        # Convert the locked sigma back to a tensor
        locked_sigma_tensor = torch.tensor(locked_sigma, dtype=torch.float32)
        return (locked_sigma_tensor,)


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

