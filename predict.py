"""
usage:  python predict.py path/to/image [more images ...]

Prints the class, the pneumonia probability, and the screening decision.
Order of checks:
  1. colour image?           -> reject, chest X-rays are greyscale
  2. (almost) blank image?   -> reject
  3. embedding far from anything seen in training (Mahalanobis > 99.5th pct of val) -> reject
  4. probability inside the refer band [99.5%-sensitivity point, 98%-specificity point]
                             -> "low confidence - refer to a clinician"
  5. otherwise NORMAL / PNEUMONIA at the cost-based threshold t*
Rejected inputs never get a pneumonia / normal answer.
"""
import os
import sys
import json
import time

import numpy as np
import cv2
from PIL import Image
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import load_model, to_tensor   # noqa: E402
from common import RES, MODELS   # noqa: E402

_state = {}


def _load():
    if not _state:
        m, cfg = load_model("final.pt")
        with open(os.path.join(RES, "p2_threshold.json")) as f:
            thr = json.load(f)
        ood = np.load(os.path.join(MODELS, "ood.npz"))
        _state.update(model=m, cfg=cfg, t=thr["threshold_camp"], band=thr["refer_band"],
                      mu=ood["mean"], prec=ood["prec"], cut=float(ood["cut"]))
    return _state


def preprocess(img):
    """PIL image -> (uint8 grey 128x128, colourfulness, grey std)"""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    colour = float(np.mean(np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 1] - rgb[..., 2])))
    g = np.asarray(img.convert("L"))
    g224 = cv2.resize(g, (224, 224), interpolation=cv2.INTER_AREA)   # same 2-step resize as training
    g128 = cv2.resize(g224, (128, 128), interpolation=cv2.INTER_AREA)
    return g128, colour, float(g.std())


@torch.no_grad()
def predict_image(img):
    s = _load()
    t0 = time.perf_counter()
    g, colour, std = preprocess(img)
    res = {"colourfulness": round(colour, 2), "grey_std": round(std, 2)}
    if colour > 6:
        return {**res, "verdict": "REJECTED - colour image, not a chest X-ray", "probability": None}
    if std < 8:
        return {**res, "verdict": "REJECTED - blank / uniform image", "probability": None}
    x = to_tensor(g[None], [0])
    e = s["model"].embed(x)
    p = float(torch.sigmoid(s["model"].fc(e)).item())
    d = e.numpy()[0] - s["mu"]
    maha = float(np.sqrt(d @ s["prec"] @ d))
    res.update(probability=round(p, 4), mahalanobis=round(maha, 1), ood_cut=round(s["cut"], 1))
    lo, hi = s["band"]
    if maha > s["cut"]:
        res["verdict"] = "REJECTED - does not look like the training X-rays, refer to a clinician"
    elif lo <= p <= hi:
        lean = "PNEUMONIA" if p >= s["t"] else "NORMAL"
        res["verdict"] = f"LOW CONFIDENCE - refer to a clinician (leans {lean})"
    elif p >= s["t"]:
        res["verdict"] = "PNEUMONIA suspected - refer"
    else:
        res["verdict"] = "NORMAL"
    res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return res


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in sys.argv[1:]:
        r = predict_image(Image.open(path))
        p = "-" if r["probability"] is None else f"{r['probability']:.3f}"
        print(f"{os.path.basename(path)}: {r['verdict']}   P(pneumonia)={p}")
        print("   ", json.dumps(r))


if __name__ == "__main__":
    main()
