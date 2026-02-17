import torch
import numpy as np

class ActivationProcessor:
    def __init__(self, color_min='#001a00', color_max='#00ff00'):
        self.color_min = color_min
        self.color_max = color_max
    
    def normalize_activations(self, activation_tensor):
        if activation_tensor.numel() == 0:
            return activation_tensor
        
        min_val = activation_tensor.min().item()
        max_val = activation_tensor.max().item()
        
        if max_val - min_val < 1e-8:
            return torch.zeros_like(activation_tensor)
        
        normalized = (activation_tensor - min_val) / (max_val - min_val)
        return normalized
    
    def flatten_conv_activations(self, conv_tensor):
        if conv_tensor.dim() == 4:
            flattened = conv_tensor.mean(dim=(2, 3))
        else:
            flattened = conv_tensor
        
        if flattened.dim() > 1:
            flattened = flattened.squeeze(0)
        
        return flattened
    
    def activation_to_green_color(self, intensity):
        intensity = np.clip(intensity, 0.0, 1.0)
        
        green_value = int(intensity * 255)
        
        hex_color = f'#00{green_value:02x}00'
        return hex_color
    
    def activation_to_rgb(self, intensity):
        intensity = np.clip(intensity, 0.0, 1.0)
        return (0, int(intensity * 255), 0)
    
    def process_layer_activations(self, activations_dict):
        processed = {}
        
        for layer_name, activation_tensor in activations_dict.items():
            if 'conv' in layer_name:
                flattened = self.flatten_conv_activations(conv_tensor=activation_tensor)
            else:
                if activation_tensor.dim() > 1:
                    flattened = activation_tensor.squeeze(0)
                else:
                    flattened = activation_tensor
            
            normalized = self.normalize_activations(activation_tensor=flattened)
            
            colors = [
                self.activation_to_green_color(intensity=val.item())
                for val in normalized
            ]
            
            processed[layer_name] = {
                'values': normalized.cpu().numpy(),
                'colors': colors,
                'shape': list(flattened.shape),
                'original_shape': list(activation_tensor.shape)
            }
        
        return processed
    
    def create_heatmap_data(self, activation_tensor):
        if activation_tensor.dim() == 4:
            batch_size, channels, height, width = activation_tensor.shape
            data = activation_tensor.squeeze(0).cpu().numpy()
        elif activation_tensor.dim() == 2:
            data = activation_tensor.cpu().numpy()
        else:
            data = activation_tensor.unsqueeze(0).cpu().numpy()
        
        return data
