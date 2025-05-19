# src/crop_api_retina_face.py

from pathlib import Path
import matplotlib.image as mpimg
from transformers import pipeline
from image_manipulation_retina_face import generate_crop

class RetinaFaceSaliencyModel:
    """
    Usa o modelo RetinaFace (py-feat/retinaface, MobileNet-0.25 backbone) 
    para detectar faces e gerar crops centrados nos pontos de maior confiança.
    """
    def __init__(self, aspectRatios=None, device: int = -1):
        # aspectRatios padrão: [16/9, 1/1, 8/7, 2/1]
        self.aspectRatios = aspectRatios or [0.56, 1.0, 1.14, 2.0]
        # device = -1 (CPU) ou índice da GPU (e.g., 0)
        self.face_detector = pipeline(
            "object-detection",
            model="py-feat/retinaface",
            device=device
        )

    def get_output(self, img_path: Path, aspectRatios=None):
        # 1) Detecta faces
        detections = self.face_detector(str(img_path))
        points = []
        for det in detections:
            box = det["box"]       # {'xmin','ymin','width','height'}
            score = det["score"]
            x0, y0, w, h = box["xmin"], box["ymin"], box["width"], box["height"]
            xc, yc = x0 + w/2, y0 + h/2
            points.append((xc, yc, score))

        if not points:
            raise RuntimeError(f"Nenhuma face detectada em {img_path}")

        # 2) Ordena por confiança
        points_sorted = sorted(points, key=lambda t: t[2], reverse=True)
        top_x, top_y, _ = points_sorted[0]
        ratios = aspectRatios or self.aspectRatios

        # 3) Carrega imagem e gera crops
        img = mpimg.imread(str(img_path))
        crops = [
            generate_crop(img, int(top_x), int(top_y), ar)
            for ar in ratios
        ]

        return {
            "salient_point": [(int(top_x), int(top_y))],
            "all_salient_points": [
                (int(x), int(y), float(s)) for x, y, s in points_sorted
            ],
            "crops": crops
        }

# Exemplo de uso em CLI ou import:
#   model = RetinaFaceSaliencyModel(device=0)
#   output = model.get_output(Path("data/img.jpg"))
