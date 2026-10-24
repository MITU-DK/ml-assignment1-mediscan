"""
Part 2, task 7: can it ship?  file size, CPU latency, and the dog / blank image test.
Latency is measured on validation images with 1 and 2 CPU threads (a camp laptop is slow).
"""
import os
import time
import urllib.request

import numpy as np
from PIL import Image
import torch

from common import MODELS, ROOT, save_json, CACHE
from engine import split_idx
import pandas as pd
import predict as P

SAMPLES = os.path.join(ROOT, "samples")
os.makedirs(SAMPLES, exist_ok=True)
DOG_URL = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"


def main():
    out = {"size_MB": {}}
    for f in ["final.pt", "resnet18_s0.pt"]:
        p = os.path.join(MODELS, f)
        if os.path.exists(p):
            out["size_MB"][f] = round(os.path.getsize(p) / 1e6, 2)

    # sample inputs
    dog = os.path.join(SAMPLES, "dog.jpg")
    if not os.path.exists(dog):
        req = urllib.request.Request(DOG_URL, headers={"User-Agent": "Mozilla/5.0 (student project)"})
        with urllib.request.urlopen(req) as r, open(dog, "wb") as f:
            f.write(r.read())
    Image.fromarray(np.full((512, 512), 255, np.uint8)).save(os.path.join(SAMPLES, "blank_white.png"))
    Image.fromarray(np.zeros((512, 512), np.uint8)).save(os.path.join(SAMPLES, "blank_black.png"))
    # the dog in greyscale is the nasty case: the colour check can't catch it
    Image.open(dog).convert("L").save(os.path.join(SAMPLES, "dog_grey.png"))
    df = pd.read_csv(os.path.join(CACHE, "split.csv"))
    _, va = split_idx(0)
    for lab in [0, 1]:
        i = va[df.label.values[va] == lab][0]
        Image.open(df.path[i]).save(os.path.join(SAMPLES, f"val_{['normal', 'pneumonia'][lab]}.jpeg"))

    out["sample_predictions"] = {}
    for f in sorted(os.listdir(SAMPLES)):
        r = P.predict_image(Image.open(os.path.join(SAMPLES, f)))
        out["sample_predictions"][f] = r
        print(f, "->", r["verdict"], r.get("probability"))

    # latency, model already loaded, includes preprocessing from a decoded image
    imgs = [Image.open(df.path[i]) for i in va[:60]]
    for im in imgs:
        im.load()
    for th in [1, 2]:
        torch.set_num_threads(th)
        P.predict_image(imgs[0])
        ts = []
        for im in imgs:
            t0 = time.perf_counter()
            P.predict_image(im)
            ts.append((time.perf_counter() - t0) * 1000)
        out[f"latency_ms_{th}_thread"] = {"mean": float(np.mean(ts)), "p95": float(np.percentile(ts, 95))}
    t0 = time.perf_counter()
    P._state.clear()
    P._load()
    out["model_load_ms"] = (time.perf_counter() - t0) * 1000
    print(out)
    # SHIP_OUT lets the latency be re-measured on another machine without overwriting
    save_json(out, os.environ.get("SHIP_OUT", "p2_ship.json"))


if __name__ == "__main__":
    main()
