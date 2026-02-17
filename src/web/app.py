from flask import Flask, render_template, jsonify, request
from pathlib import Path
import torch

from src.web import config
from src.web.model_loader import ModelLoader
from src.web.sample_manager import SampleManager
from src.web.visualizer import Visualizer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

model_loader = ModelLoader(model_path=config.MODEL_PATH)
sample_manager = SampleManager(
    num_samples=config.NUM_SAMPLES,
    samples_folder=config.SAMPLES_FOLDER
)
visualizer = Visualizer()

@app.route('/')
def index():
    samples = sample_manager.get_samples_metadata()
    model_status = model_loader.get_model_status()
    return render_template(
        'index.html',
        samples=samples,
        model_status=model_status,
    )

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'model_type': model_loader.model_type,
        'num_samples': len(sample_manager.get_samples_metadata()),
    })

@app.route('/api/visualize/<int:image_id>')
def api_visualize(image_id):
    try:
        image_data = sample_manager.get_sample_by_id(sample_id=image_id)
        if image_data is None:
            return jsonify({'error': 'Image not found'}), 404

        preprocessed_image = sample_manager.preprocess_image(
            image=image_data['image']
        )
        activations = model_loader.get_activations(
            input_tensor=preprocessed_image
        )
        network_graph_html = visualizer.create_network_graph(
            activations=activations
        )
        heatmaps_html = visualizer.create_heatmaps(activations=activations)

        return jsonify({
            'image_id': image_id,
            'image_path': image_data['path'],
            'class_label': image_data['label'],
            'network_graph': network_graph_html,
            'heatmaps': heatmaps_html,
            'model_status': model_loader.get_model_status(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualize-multi', methods=['POST'])
def api_visualize_multi():
    try:
        data = request.get_json()
        image_ids = data.get('image_ids', [])

        if not image_ids:
            return jsonify({'error': 'image_ids is required'}), 400

        all_activations = []
        image_paths = []
        class_labels = []

        for img_id in image_ids:
            image_data = sample_manager.get_sample_by_id(sample_id=img_id)
            if image_data is None:
                return jsonify({'error': f'Image {img_id} not found'}), 404

            preprocessed = sample_manager.preprocess_image(
                image=image_data['image']
            )
            activations = model_loader.get_activations(
                input_tensor=preprocessed
            )
            all_activations.append(activations)
            image_paths.append(image_data['path'])
            class_labels.append(image_data['label'])

        averaged = {}
        for layer_name in all_activations[0]:
            stacked = torch.stack(
                [a[layer_name] for a in all_activations]
            )
            averaged[layer_name] = stacked.mean(dim=0)

        network_graph_html = visualizer.create_network_graph(
            activations=averaged
        )
        heatmaps_html = visualizer.create_heatmaps(activations=averaged)

        return jsonify({
            'image_ids': image_ids,
            'image_paths': image_paths,
            'class_labels': class_labels,
            'network_graph': network_graph_html,
            'heatmaps': heatmaps_html,
            'model_status': model_loader.get_model_status(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/activations', methods=['POST'])
def get_activations_api():
    try:
        data = request.get_json()
        image_id = data.get('image_id')

        if image_id is None:
            return jsonify({'error': 'image_id is required'}), 400

        image_data = sample_manager.get_sample_by_id(sample_id=image_id)
        if image_data is None:
            return jsonify({'error': 'Image not found'}), 404

        preprocessed_image = sample_manager.preprocess_image(
            image=image_data['image']
        )
        activations = model_loader.get_activations(
            input_tensor=preprocessed_image
        )

        processed_activations = {}
        for layer_name, activation_tensor in activations.items():
            processed_activations[layer_name] = {
                'shape': list(activation_tensor.shape),
                'values': activation_tensor.cpu().numpy().tolist(),
            }

        return jsonify({
            'image_id': image_id,
            'activations': processed_activations,
            'model_status': model_loader.get_model_status(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload-model', methods=['POST'])
def upload_model():
    try:
        if 'model_file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400

        file = request.files['model_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        extension = Path(file.filename).suffix.lower()
        if extension not in config.ALLOWED_MODEL_EXTENSIONS:
            return jsonify({
                'error': (
                    f'Invalid file type "{extension}". '
                    f'Allowed: {", ".join(config.ALLOWED_MODEL_EXTENSIONS)}'
                )
            }), 400

        save_path = config.UPLOAD_FOLDER / file.filename
        file.save(str(save_path))

        model_loader.load_model_from_path(model_path=save_path)

        return jsonify({
            'message': f'Model "{file.filename}" loaded successfully',
            'model_status': model_loader.get_model_status(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"Starting Flask app on {config.FLASK_HOST}:{config.FLASK_PORT}")
    print(f"Model status: {model_loader.get_model_status()}")
    print(f"Samples loaded: {len(sample_manager.get_samples_metadata())}")
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
