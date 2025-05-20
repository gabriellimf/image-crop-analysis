# src/crop_api_retina_face.py
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
        detections = RetinaFace.detect_faces(str(img_path))            # dict com "score" :contentReference[oaicite:1]{index=1}
        pts = [                                                       # (xc, yc, score)
            ((x1 + x2) / 2, (y1 + y2) / 2, det["score"])
            for det in detections.values()
            for x1, y1, x2, y2 in [det["facial_area"]]
        ]

        img = mpimg.imread(str(img_path))
        h, w = img.shape[:2]

        if not pts:                                                   # fallback se nada detectado
            pts = [(w/2, h/2, 1.0)]

        # ------------------------------------------------------------------
        #  ⚖️  Estratégia de desempate: escolher aleatoriamente entre Top-2
        # ------------------------------------------------------------------
        pts_sorted = sorted(pts, key=lambda p: p[2], reverse=True)    # ordem por score
        tie_pool   = pts_sorted[:2]                                   # Top-2  :contentReference[oaicite:2]{index=2}
        top_x, top_y, _ = random.choice(tie_pool)                     # 50/50  :contentReference[oaicite:3]{index=3}
        ratios = aspectRatios or self.aspectRatios or [0.56, 1.0, 1.14, 2.0]
        crops = [generate_crop(img, int(top_x), int(top_y), ar) for ar in ratios]
        return {
            "salient_point":      [(int(top_x), int(top_y))],
            "all_salient_points": [(int(x), int(y), float(s)) for x, y, s in pts_sorted],
            "crops": crops,
        }
        
    def plot_img_crops(self, img_path, topK=1, **kwargs):
        out = self.get_output(img_path)
        n = len(out["all_salient_points"])
        topK = min(topK, n)  # garante que não haverá IndexError
        super().plot_img_crops(img_path, topK=topK, **kwargs)
