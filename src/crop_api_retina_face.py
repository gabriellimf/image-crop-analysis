"""
Versão aprimorada do wrapper RetinaFace para o pipeline de cropping.

Mudanças principais para mitigar viés:
1. **Ranking equitativo** – o score de confiança da RetinaFace é
   normalizado pela área da *bounding‑box* (score/area) antes da ordenação.
2. **Desempate Top‑2 aleatório** – quando há pelo menos duas faces, o
   ponto central é escolhido aleatoriamente entre as duas de maior score
   normalizado.
3. **Sem reflexo de simetria** – métodos de plotagem chamam a versão da
   superclasse com `checkSymmetry=False`, evitando crops espelhados que
   podiam favorecer rostos à direita.
4. **Proteção de `topK`** – garante que `topK` não exceda o número de
   faces detectadas (evita `IndexError`).

Este arquivo substitui o `src/crop_api_retina_face.py` anterior; nenhum
notebook precisa ser alterado.
"""

from PIL import Image

from pathlib import Path
import random
from typing import List, Tuple

import matplotlib.image as mpimg
from retinaface import RetinaFace

from crop_api import ImageSaliencyModel, generate_crop

Point = Tuple[float, float, float]  # (x, y, score)

__all__ = ["RetinaFaceSaliencyModel"]


class RetinaFaceSaliencyModel(ImageSaliencyModel):
    """Gera crops centrados em faces detectadas pelo RetinaFace.

    Parâmetros
    ----------
    aspectRatios : list[float], opcional
        Razões de aspecto desejadas. Default: ``[0.56, 1.0, 1.14, 2.0]``.
    """

    def __init__(self, aspectRatios: List[float] | None = None):
        # A superclasse exige caminhos para o binário C++; enviamos strings vazias
        super().__init__(crop_binary_path="", crop_model_path="", aspectRatios=aspectRatios)

    # ------------------------------------------------------------------
    # 1. Detecção de faces + escolha do ponto saliente
    # ------------------------------------------------------------------
    def _detect_faces(self, img_path: Path) -> List[Point]:
        """Executa RetinaFace e devolve [(xc, yc, score_norm)]."""
        detections = RetinaFace.detect_faces(str(img_path)) or {}
        pts: List[Point] = []
        for det in detections.values():
            x1, y1, x2, y2 = det["facial_area"]
            area = max((x2 - x1) * (y2 - y1), 1.0)  # evita div/0
            score_norm = det["score"] / area       # normaliza por área
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            pts.append((xc, yc, score_norm))
        return pts

    def get_output(self, img_path: Path, aspectRatios=None):
        pts = self._detect_faces(img_path)
        img = mpimg.imread(str(img_path))
        h, w = img.shape[:2]

        # fallback quando nenhuma face é detectada
        if not pts:
            pts = [(w / 2, h / 2, 1.0)]

        # ordena por score normalizado (desc) e aplica desempate aleatório
        pts_sorted = sorted(pts, key=lambda p: p[2], reverse=True)
        tie_pool = pts_sorted[:2]  # até 2 melhores
        top_x, top_y, _ = random.choice(tie_pool)

        ratios = aspectRatios or self.aspectRatios or [0.56, 1.0, 1.14, 2.0]
        crops = [generate_crop(img, int(top_x), int(top_y), ar) for ar in ratios]

        return {
            "salient_point": [(int(top_x), int(top_y))],
            "all_salient_points": [(int(x), int(y), float(s)) for x, y, s in pts_sorted],
            "crops": crops,
        }

    # ------------------------------------------------------------------
    # 2. Plotagem – desliga simetria e previne IndexError de topK
    # ------------------------------------------------------------------
    def plot_img_crops(self, img_path, topK: int = 1, **kwargs):
        out = self.get_output(img_path)
        n_pts = len(out["all_salient_points"])
        super().plot_img_crops(
            img_path,
            topK=min(topK, n_pts),
            checkSymmetry=False,      # 🔕 não gera crop espelhado
            **kwargs,
        )

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
