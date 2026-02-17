import plotly.graph_objects as go
import numpy as np
from src.web.activation_processor import ActivationProcessor

LAYER_SORT_KEY = {
    'conv1': 0,
    'conv2': 1,
    'conv3': 2,
    'conv4': 3,
    'fc_1': 10,
    'fc_2': 11,
    'fc_3': 12,
    'fc_out': 13,
    'kan_inner': 20,
    'kan_outer': 21,
}

LAYER_LABELS = {
    'conv1': 'Conv 1',
    'conv2': 'Conv 2',
    'conv3': 'Conv 3',
    'conv4': 'Conv 4',
    'fc_1': 'FC 1',
    'fc_2': 'FC 2',
    'fc_3': 'FC 3',
    'fc_out': 'FC Out',
    'kan_inner': 'KAN Inner',
    'kan_outer': 'KAN Outer',
}


def _sorted_layer_names(activations):
    return sorted(
        activations.keys(),
        key=lambda name: LAYER_SORT_KEY.get(name, 99),
    )


def _layer_label(name):
    return LAYER_LABELS.get(name, name)


class Visualizer:
    def __init__(self):
        self.processor = ActivationProcessor()

    def create_network_graph(self, activations):
        processed = self.processor.process_layer_activations(
            activations_dict=activations
        )
        layer_names = _sorted_layer_names(processed)

        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        node_sizes = []

        tick_labels = []
        tick_vals = []

        for idx, layer_name in enumerate(layer_names):
            y_pos = len(layer_names) - 1 - idx
            tick_labels.append(_layer_label(name=layer_name))
            tick_vals.append(y_pos)

            layer_data = processed[layer_name]
            values = layer_data['values']
            colors = layer_data['colors']
            num_neurons = len(values)

            max_neurons_display = 50
            if num_neurons > max_neurons_display:
                step = num_neurons // max_neurons_display
                display_indices = list(range(0, num_neurons, step))[
                    :max_neurons_display
                ]
            else:
                display_indices = list(range(num_neurons))

            for pos, neuron_idx in enumerate(display_indices):
                if len(display_indices) > 1:
                    x_pos = (pos / (len(display_indices) - 1)) - 0.5
                else:
                    x_pos = 0
                node_x.append(x_pos)
                node_y.append(y_pos)
                node_colors.append(colors[neuron_idx])
                node_text.append(
                    f'{_layer_label(name=layer_name)}<br>'
                    f'Neuron {neuron_idx}<br>Value: {values[neuron_idx]:.4f}'
                )
                node_sizes.append(10 + values[neuron_idx] * 5)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='white'),
            ),
            text=node_text,
            hoverinfo='text',
            showlegend=False,
        ))

        has_conv = any(n.startswith('conv') for n in layer_names)
        has_kan = any(n.startswith('kan') for n in layer_names)
        has_fc = any(n.startswith('fc') for n in layer_names)

        if has_conv and has_kan:
            title = 'KAN-CNN Activations'
        elif has_conv and has_fc:
            title = 'Baseline CNN Activations'
        elif has_kan:
            title = 'Plain KAN Activations'
        elif has_fc:
            title = 'Plain MLP Activations'
        else:
            title = 'Network Activations'

        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(
                showgrid=True,
                zeroline=False,
                showticklabels=True,
                ticktext=tick_labels,
                tickvals=tick_vals,
            ),
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#0a0a0a',
            font=dict(color='white'),
            height=max(400, 100 * len(layer_names)),
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def create_heatmaps(self, activations):
        processed = self.processor.process_layer_activations(
            activations_dict=activations
        )
        layer_names = _sorted_layer_names(processed)

        heatmaps_html = []

        for layer_name in layer_names:
            layer_data = processed[layer_name]
            values = layer_data['values']
            values_2d = values.reshape(1, -1)

            fig = go.Figure(data=go.Heatmap(
                z=values_2d,
                colorscale=[
                    [0, '#001a00'],
                    [0.25, '#003300'],
                    [0.5, '#006600'],
                    [0.75, '#00cc00'],
                    [1, '#00ff00'],
                ],
                colorbar=dict(
                    title=dict(text='Activation', side='right'),
                    tickmode='linear',
                    tick0=0,
                    dtick=0.25,
                ),
                hovertemplate=(
                    'Neuron: %{x}<br>Activation: %{z:.4f}<extra></extra>'
                ),
            ))

            fig.update_layout(
                title=f'{_layer_label(name=layer_name)} Activations',
                xaxis=dict(title='Neuron Index'),
                yaxis=dict(showticklabels=False),
                height=200,
                margin=dict(l=50, r=50, t=50, b=50),
                plot_bgcolor='#0a0a0a',
                paper_bgcolor='#0a0a0a',
                font=dict(color='white'),
            )

            heatmaps_html.append(
                fig.to_html(full_html=False, include_plotlyjs='cdn')
            )

        return '\n'.join(heatmaps_html)

    def create_activation_summary(self, activations):
        processed = self.processor.process_layer_activations(
            activations_dict=activations
        )

        summary = {}
        for layer_name, layer_data in processed.items():
            values = layer_data['values']
            summary[layer_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'num_neurons': len(values),
            }

        return summary
