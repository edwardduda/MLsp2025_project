import torch
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path
import random

MNIST_LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]


class SampleManager:
    def __init__(self, num_samples, samples_folder):
        self.num_samples = num_samples
        self.samples_folder = Path(samples_folder)
        self.samples = []

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ])

        self._load_samples()

    def _load_samples(self):
        try:
            print("Loading MNIST test dataset...")
            dataset = datasets.MNIST(
                root='data', train=False, download=True
            )

            total_size = len(dataset)
            indices = random.sample(
                range(total_size), min(self.num_samples, total_size)
            )

            self.samples_folder.mkdir(parents=True, exist_ok=True)

            for idx, dataset_idx in enumerate(indices):
                image, label = dataset[dataset_idx]

                image_filename = f'sample_{idx}.png'
                image_path = self.samples_folder / image_filename
                image.save(image_path)

                self.samples.append({
                    'id': idx,
                    'path': f'static/samples/{image_filename}',
                    'label': MNIST_LABELS[label],
                    'image': image,
                })

            print(f"Loaded {len(self.samples)} MNIST samples")

        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating placeholder samples...")
            self._create_placeholder_samples()

    def _create_placeholder_samples(self):
        for idx in range(min(5, self.num_samples)):
            image = Image.new('L', (28, 28), color=0)

            image_filename = f'placeholder_{idx}.png'
            image_path = self.samples_folder / image_filename
            image.save(image_path)

            self.samples.append({
                'id': idx,
                'path': f'static/samples/{image_filename}',
                'label': '0',
                'image': image,
            })

        print(f"Created {len(self.samples)} placeholder samples")
    
    def get_samples_metadata(self):
        return [
            {
                'id': sample['id'],
                'path': sample['path'],
                'label': sample['label']
            }
            for sample in self.samples
        ]
    
    def get_sample_by_id(self, sample_id):
        for sample in self.samples:
            if sample['id'] == sample_id:
                return sample
        return None
    
    def preprocess_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL Image or path string")
        
        tensor = self.transform(image)
        return tensor
