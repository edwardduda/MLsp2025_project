import torch
from pathlib import Path

from src.models.cnn import BaselineCNN
from src.models.kan_cnn import KANCNN
from src.models.mlp import PlainMLP
from src.models.kan import PlainKAN


def _detect_model_type(state_dict):
    keys = set(state_dict.keys())
    has_conv = any(k.startswith("conv1.") for k in keys)
    has_kan = any(k.startswith("kan1.") for k in keys)
    has_ffn = any(k.startswith("ffn.") for k in keys)

    if has_conv and has_kan:
        return "kan_cnn"
    if has_conv and has_ffn:
        return "baseline_cnn"
    if has_kan and not has_conv:
        return "plain_kan"
    if has_ffn and not has_conv:
        return "plain_mlp"

    raise ValueError("Cannot determine model type from state dict keys")


def _build_model(model_type):
    if model_type == "kan_cnn":
        return KANCNN(
            n_classes=10,
            input_shape=(1, 28, 28),
            ch1=32,
            ch2=64,
            ch3=128,
            ch4=256,
            kan_1=64,
            kan_2=32,
            spline_cp=7,
            spline_deg=3,
            range_min=-3.0,
            range_max=10.0,
            dropout_p=0.1,
        )
    if model_type == "baseline_cnn":
        return BaselineCNN(n_classes=10, dropout_p=0.1, in_channels=1)
    if model_type == "plain_mlp":
        return PlainMLP(n_classes=10, dropout_p=0.1, input_dim=784)
    if model_type == "plain_kan":
        return PlainKAN(
            n_classes=10,
            input_dim=784,
            kan_1=128,
            kan_2=64,
            spline_cp=7,
            spline_deg=3,
            range_min=-3.0,
            range_max=10.0,
            dropout_p=0.1,
        )
    raise ValueError(f"Unknown model type: {model_type}")


class ModelLoader:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.model_type = None
        self.activations = {}
        self.hooks = []
        self.model_loaded = False

        self._load_model()
        self._register_hooks()

    def _load_model(self):
        try:
            if self.model_path.exists():
                state_dict = torch.load(self.model_path, map_location='cpu')
                self.model_type = _detect_model_type(state_dict=state_dict)
                self.model = _build_model(model_type=self.model_type)
                self.model.load_state_dict(state_dict)
                self.model_loaded = True
                print(f"Model loaded from {self.model_path} (type: {self.model_type})")
            else:
                print(f"Warning: Model file not found at {self.model_path}")
                print("Using untrained KANCNN for demonstration")
                self.model_type = "kan_cnn"
                self.model = _build_model(model_type=self.model_type)
                self.model_loaded = False

            self.model.eval()

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach().clone()
        return hook

    def _register_hooks(self):
        if self.model is None:
            return

        hook_targets = self._get_hook_targets()
        for name, module in hook_targets:
            self.hooks.append(
                module.register_forward_hook(self._get_activation(name))
            )

        print(f"Registered {len(self.hooks)} activation hooks for {self.model_type}")

    def _get_hook_targets(self):
        if self.model_type == "kan_cnn":
            return [
                ('conv1', self.model.conv1),
                ('conv2', self.model.conv2),
                ('conv3', self.model.conv3),
                ('conv4', self.model.conv4),
                ('kan_inner', self.model.kan1),
                ('kan_outer', self.model.kan2),
            ]
        if self.model_type == "baseline_cnn":
            # ffn: [0]Linear [1]ReLU [2]Dropout [3]Linear [4]ReLU [5]Dropout [6]Linear
            return [
                ('conv1', self.model.conv1),
                ('conv2', self.model.conv2),
                ('conv3', self.model.conv3),
                ('conv4', self.model.conv4),
                ('fc_1', self.model.ffn[1]),
                ('fc_2', self.model.ffn[4]),
                ('fc_out', self.model.ffn[6]),
            ]
        if self.model_type == "plain_mlp":
            # ffn: [0]Linear [1]BN [2]ReLU [3]Drop [4]Linear [5]BN [6]ReLU
            #      [7]Drop [8]Linear [9]BN [10]ReLU [11]Drop [12]Linear
            return [
                ('fc_1', self.model.ffn[2]),
                ('fc_2', self.model.ffn[6]),
                ('fc_3', self.model.ffn[10]),
                ('fc_out', self.model.ffn[12]),
            ]
        if self.model_type == "plain_kan":
            return [
                ('kan_inner', self.model.kan1),
                ('kan_outer', self.model.kan2),
            ]
        return []

    def get_activations(self, input_tensor):
        self.activations = {}

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            self.model(input_tensor)

        return self.activations.copy()

    def load_model_from_path(self, model_path):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        self.model_path = Path(model_path)
        self._load_model()
        self._register_hooks()

    def get_model_status(self):
        return {
            'loaded': self.model_loaded,
            'path': str(self.model_path),
            'exists': self.model_path.exists(),
            'model_type': self.model_type,
            'message': 'Model loaded successfully' if self.model_loaded else 'Using untrained model'
        }

    def __del__(self):
        for hook in self.hooks:
            hook.remove()
