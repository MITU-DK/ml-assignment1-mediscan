"""
Downloads both Kaggle datasets with kagglehub (public, no key needed at time of writing).

  data/chest_xray      -> Kermany et al. pediatric pneumonia set (Rohit's download)
  data/external_raw    -> COVID-19 Radiography Database. Only the Normal and
                          Viral Pneumonia folders are used, and only by external_eval.py.

This script does not open or look at the external images, it only downloads.
"""
import os
import kagglehub

from common import DATA

primary = os.path.join(DATA, "chest_xray_raw")
external = os.path.join(DATA, "external_raw")
link = os.path.join(DATA, "chest_xray")
ext_name = "COVID-19_Radiography_Dataset"


def find_dir(root, must_have):
    for dirpath, dirnames, _ in os.walk(root):
        if all(os.path.isdir(os.path.join(dirpath, m)) for m in must_have):
            return dirpath
    return None


# on Kaggle, with both datasets added as inputs, just link to them (no download)
if os.path.isdir("/kaggle/input"):
    p = find_dir("/kaggle/input", ["train/NORMAL", "test/PNEUMONIA"])
    e = find_dir("/kaggle/input", ["Normal/images", "Viral Pneumonia/images"])
    if p and not os.path.exists(link):
        os.symlink(p, link)
    if e and not os.path.exists(os.path.join(external, ext_name)):
        os.makedirs(external, exist_ok=True)
        os.symlink(e, os.path.join(external, ext_name))
    print("kaggle inputs:", p, e)

if not os.path.exists(link):
    # inside a Kaggle notebook kagglehub may ignore output_dir and return its own path,
    # so look for the folders under whatever path it gives back
    got = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia", output_dir=primary)
    p = find_dir(got, ["train/NORMAL", "test/PNEUMONIA"]) or find_dir(primary, ["train/NORMAL", "test/PNEUMONIA"])
    os.symlink(p, link)

if not os.path.exists(os.path.join(external, ext_name)):
    got = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database", output_dir=external)
    e = find_dir(got, ["Normal/images", "Viral Pneumonia/images"])
    if e and not os.path.exists(os.path.join(external, ext_name)):
        os.makedirs(external, exist_ok=True)
        os.symlink(e, os.path.join(external, ext_name))

assert os.path.isdir(os.path.join(link, "train", "NORMAL")), "primary dataset not found"
assert os.path.isdir(os.path.join(external, ext_name, "Normal", "images")), "external dataset not found"
print("data ready in", DATA)
