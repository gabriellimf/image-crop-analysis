# src/crop_api_retina_face.py
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.image as mpimg
import random
from retinaface import RetinaFace
from crop_api import ImageSaliencyModel, generate_crop

class RetinaFaceSaliencyModel(ImageSaliencyModel):
    def __init__(self, aspectRatios=None):
        # Chama o construtor da classe-mãe, mas passa dummy paths
        super().__init__(crop_binary_path="", crop_model_path="", aspectRatios=aspectRatios)

    # sobrescreve apenas a lógica de saída
    def get_output(self, img_path: Path, aspectRatios=None):
        detections = RetinaFace.detect_faces(str(img_path))
        pts = [
            ((x1 + x2) / 2, (y1 + y2) / 2, det["score"])
            for det in detections.values()
            for x1, y1, x2, y2 in [det["facial_area"]]
        ]
        img = mpimg.imread(str(img_path))
        h, w = img.shape[:2]
        if not pts:
            pts = [(w/2, h/2, 1.0)]
        top_x, top_y, _ = max(pts, key=lambda p: p[2])
        ratios = aspectRatios or self.aspectRatios or [0.56, 1.0, 1.14, 2.0]
        crops = [generate_crop(img, int(top_x), int(top_y), ar) for ar in ratios]
        return {
            "salient_point": [(int(top_x), int(top_y))],
            "all_salient_points": [(int(x), int(y), float(s)) for x, y, s in pts],
            "crops": crops,
        }
        
    def plot_img_crops(self, img_path, topK=1, **kwargs):
        out = self.get_output(img_path)
        n = len(out["all_salient_points"])
        topK = min(topK, n)  # garante que não haverá IndexError
        super().plot_img_crops(img_path, topK=topK, **kwargs)

    def save_crops(self, img_path: Path, output_dir: Path, prefix="crop", image_format="JPEG"):
        """Gera e salva os crops como arquivos de imagem.

        Args:
            img_path: caminho da imagem original
            output_dir: diretório para salvar os crops
            prefix: prefixo para os arquivos
            image_format: formato da imagem (ex: "JPEG", "PNG")
        """
        output = self.get_output(img_path)
        crops = output["crops"]
        img = Image.open(str(img_path))
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, rect in enumerate(crops):
            left, top, width, height = rect
            box = (left, top, left + width, top + height)
            cropped = img.crop(box)
            crop_path = output_dir / f"{prefix}_{i}.{image_format.lower()}"
            cropped.save(crop_path, format=image_format)