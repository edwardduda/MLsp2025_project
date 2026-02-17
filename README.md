# MLsp2025 Project: Kolmogorov-Arnold Networks (KAN)

A research project exploring Kolmogorov-Arnold Networks (KAN) with interactive visualization capabilities. This repository includes model implementations, training scripts, and a Flask web application for visualizing neuron activations.

## Table of Contents

- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [Flask Web Application](#flask-web-application)
- [Training Models](#training-models)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)

## Features

### Research & Experimentation
- Multiple model implementations: KAN, CNN, MLP variants
- Jupyter notebooks for experimentation and proof-of-concept
- Training scripts with history tracking
- Model comparison capabilities

### Interactive Visualization
- **Interactive Network Graph**: Visualize all layers with neurons colored by activation intensity (green gradient)
- **Layer Heatmaps**: View detailed activation patterns for each network layer
- **Sample Image Gallery**: Select from Tiny ImageNet dataset samples
- **Real-time Inference**: See activations update for each input image
- **Model Status Indicator**: Shows whether a trained model is loaded or demo mode is active

## Installation & Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd MLsp2025_project
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Running the Flask Web Application

**Option 1: Using the startup script (Recommended)**
```bash
./scripts/run_flask.sh
```

**Option 2: Direct Python execution**
```bash
python3 -m src.web.app
```

The app will start on `http://localhost:5000`

### Adding Your Trained Model

When you have a trained model:

```bash
# Create saved_models directory if it doesn't exist
mkdir -p saved_models

# Save your trained model in your training script/notebook:
# torch.save(model.state_dict(), 'saved_models/kan_model.pth')

# Copy or move your model file
cp /path/to/your/model.pth saved_models/kan_model.pth
```

Then restart the Flask app to load the trained model.

### Training Models

Run the training script:
```bash
python3 -m src.training.train
```

This will train models and save checkpoints to the `saved_models/` directory.

## Flask Web Application

### Main Page (Gallery)
- Browse 25 sample images from Tiny ImageNet or MNIST
- Click any image to visualize its activations

### Visualization Page
- **Network Graph Tab**: Interactive graph showing all layers
  - Neurons colored by activation intensity (green scale)
  - Hover to see activation values
  - Zoom and pan controls
  
- **Heatmaps Tab**: Layer-by-layer activation heatmaps
  - Detailed view of each layer's activations
  - Color scale from dark green (low) to bright green (high)

### Color Legend

- 🟢 Bright Green (#00ff00): High activation
- 🟢 Medium Green (#006600): Medium activation  
- 🟢 Dark Green (#001a00): Low activation

## Training Models

The repository includes training scripts for multiple model architectures:

- **Baseline CNN**: Standard convolutional neural network
- **KAN-CNN**: Hybrid model combining CNN with KAN layers
- **Plain MLP**: Multi-layer perceptron baseline
- **Plain KAN**: Pure KAN architecture

Training results are saved to the `runs/` directory with:
- Model checkpoints (`.pt` files)
- Training history (`.json` files)
- Early stopping information

## Project Structure

```
MLsp2025_project/
├── README.md                    # This file
├── .gitignore                   # Git exclusions
├── requirements.txt             # Python dependencies
│
├── notebooks/                   # Jupyter notebooks
│   ├── ML445-2025.ipynb
│   └── KAN_vs_MLP_Proof_of_concept.ipynb
│
├── assets/                      # Project images and documentation assets
│   ├── PlotABsplineExample_01.png
│   └── neural-scaling-laws.jpg
│
├── src/                         # Source code
│   ├── models/                  # Model implementations
│   │   ├── activations.py       # BSplineActivation
│   │   ├── cnn.py               # BaselineCNN
│   │   ├── kan.py               # KAN, PlainKAN
│   │   ├── kan_cnn.py           # KANCNN
│   │   ├── kan_layer.py         # KANLayer
│   │   ├── mlp.py               # PlainMLP
│   │   ├── config.py            # Model Config class
│   │   └── data_pipeline.py    # Data processing utilities
│   │
│   ├── training/                # Training scripts
│   │   └── train.py
│   │
│   └── web/                     # Flask web application
│       ├── app.py               # Main Flask app
│       ├── config.py            # Flask configuration
│       ├── model_loader.py      # Model loading with activation hooks
│       ├── activation_processor.py  # Activation processing
│       ├── sample_manager.py    # Sample image management
│       ├── visualizer.py        # Graph and heatmap generation
│       ├── static/              # Static assets (CSS, JS, samples)
│       └── templates/           # HTML templates
│
├── scripts/                     # Utility scripts
│   └── run_flask.sh            # Flask startup script
│
├── saved_models/                # Trained model weights
│   └── *.pt, *.json
│
├── runs/                        # Training experiment outputs
│
└── data/                        # Dataset storage (MNIST, etc.)
```

## Architecture

The KAN network visualized in the Flask app has the following architecture:

- **Conv L1**: 3 → 64 channels (3x3 kernel)
- **Conv L2**: 64 → 256 channels (3x3 kernel) 
- **Conv L3**: 256 → 4096 channels (3x3 kernel)
- **KAN Inner**: Flattened → 15 functions with B-spline activation
- **KAN Outer**: 15 → 19 functions with B-spline activation

## Configuration

Edit `src/web/config.py` to customize Flask app settings:

```python
MODEL_PATH = BASE_DIR / 'saved_models' / 'kan_model.pth'  # Path to model file
NUM_SAMPLES = 25                                           # Number of sample images
IMAGE_SIZE = (64, 64)                                      # Input image dimensions
FLASK_PORT = 5000                                          # Flask server port
```

## API Reference

### GET `/`
Main page with image gallery

### GET `/visualize/<image_id>`
Visualization page for a specific image

### POST `/api/activations`
JSON API endpoint for activation data

**Request:**
```json
{
  "image_id": 0
}
```

**Response:**
```json
{
  "image_id": 0,
  "activations": {
    "conv_l1": {
      "shape": [1, 64, 64, 64],
      "values": [[...]]
    }
  },
  "model_status": {
    "loaded": true,
    "message": "Model loaded successfully"
  }
}
```

**Using curl:**
```bash
curl -X POST http://localhost:5000/api/activations \
  -H "Content-Type: application/json" \
  -d '{"image_id": 0}'
```

## Troubleshooting

### Model Not Loading
- Ensure the model file exists at `saved_models/kan_model.pth`
- Check that the model architecture matches (input channels, layer sizes)
- Verify the model state dict keys match the KAN definition

### No Sample Images Displayed
- First run downloads Tiny ImageNet (requires internet connection)
- Falls back to placeholder images if download fails
- Check `src/web/static/samples/` directory for cached images

### Visualization Not Displaying
- Clear browser cache and refresh
- Check browser console for JavaScript errors
- Ensure Plotly CDN is accessible (requires internet)

### Port Already in Use
- Change `FLASK_PORT` in `src/web/config.py`
- Or kill the process using port 5000: `lsof -ti:5000 | xargs kill`

### Import Errors
- Ensure you're running from the repository root
- Use `python3 -m src.web.app` instead of `python3 src/web/app.py`
- Check that `__init__.py` files exist in all `src/` subdirectories

## Tech Stack

### Backend
- **Flask 2.3+**: Web framework
- **PyTorch 2.0+**: ML framework and model inference
- **Plotly 5.18+**: Interactive visualizations
- **NumPy 1.24+**: Numerical operations
- **Pillow 10.0+**: Image processing

### Data
- **HuggingFace Datasets 2.16+**: Dataset loading (Tiny ImageNet, MNIST)
- **torchvision 0.15+**: Dataset utilities and transforms

### Frontend
- **Bootstrap 5.3**: UI framework
- **Vanilla JavaScript**: Interactive controls
- **Plotly.js**: Client-side visualization rendering

## License

Part of ML445 2025 Course Project

---

For questions or issues, please open an issue on the repository.
