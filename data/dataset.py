import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class EmbeddingDataset(Dataset):
    """
    Dataset that loads pre-extracted SigLIP and T5 embeddings alongside original images.

    Expected directory structure for embeddings:
        embedding_dir/
            {task_name}/
                episode_XXXXXX/
                    image_X.0.pt   # dict with 'z_img' (D,) and 'z_txt' (T, C)

    Original images:
        image_dir/
            {task_name}/videos/chunk-000/observation.images.image_top/
                episode_XXXXXX/
                    image_X.0.jpg
    """

    def __init__(
        self,
        data_dir: str,
        image_dir: str = None,
        resolution: int = 256,
    ):
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.resolution = resolution

        # Find all embedding files
        self.samples = sorted(glob.glob(os.path.join(data_dir, "**", "*.pt"), recursive=True))

        if len(self.samples) == 0:
            raise ValueError(f"No .pt files found in {data_dir}")

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _get_image_path(self, embedding_path: str) -> str:
        """Derive original image path from embedding path."""
        # embedding: data_dir/{task}/episode_XXXXXX/image_X.0.pt
        # image: image_dir/{task}/videos/chunk-000/observation.images.image_top/episode_XXXXXX/image_X.0.jpg
        rel = os.path.relpath(embedding_path, self.data_dir)
        parts = rel.split(os.sep)
        task_name = parts[0]
        episode = parts[1]
        filename = os.path.splitext(parts[2])[0] + ".jpg"
        return os.path.join(
            self.image_dir, task_name,
            "videos", "chunk-000", "observation.images.image_top",
            episode, filename,
        )

    def __getitem__(self, idx: int) -> dict:
        emb_path = self.samples[idx]
        data = torch.load(emb_path, map_location="cpu", weights_only=True)

        z_img = data["z_img"]  # (D,)
        z_txt = data["z_txt"]  # (T, C)

        result = {"z_img": z_img, "z_txt": z_txt}

        # Load original image for VAE encoding
        if self.image_dir is not None:
            img_path = self._get_image_path(emb_path)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image)
                result["image"] = image
            else:
                # Fallback: create dummy image (should not happen with correct data)
                result["image"] = torch.zeros(3, self.resolution, self.resolution)

        return result
