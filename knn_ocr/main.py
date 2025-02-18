import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.measure import label, regionprops
from pathlib import Path


def extract_features(region):
    img = region.image
    h, w = img.shape

    area = img.sum() / img.size
    perimeter = region.perimeter / img.size
    wh_ratio = h / w if w > 0 else 0

    cy, cx = region.local_centroid
    cy /= h
    cx /= w

    euler = region.euler_number
    central_region = img[int(0.45 * h):int(0.55 * h), int(0.45 * w):int(0.55 * w)]
    kl = 3 * central_region.sum() / img.size if img.size > 0 else 0
    kls = 2 * central_region.sum() / img.size if img.size > 0 else 0
    eccentricity = region.eccentricity * 8 if hasattr(region, 'eccentricity') else 0

    have_v1 = (np.mean(img, axis=0) > 0.87).sum() > 2
    have_g1 = (np.mean(img, axis=1) > 0.85).sum() > 2
    have_g2 = (np.mean(img, axis=1) > 0.5).sum() > 2

    hole_size = img.sum() / region.filled_area if region.filled_area > 0 else 0
    solidity = region.solidity * 2 if hasattr(region, 'solidity') else 0

    return np.array([
        area, perimeter, cy, cx, euler, eccentricity, have_v1 * 3,
        hole_size, have_g1 * 4, have_g2 * 5, kl, wh_ratio, kls, solidity
    ])


def load_training_data(training_dir):
    training_dir = Path(training_dir)
    features, labels_list = [], []

    for label_idx, label_name in enumerate(training_dir.iterdir()):
        if not label_name.is_dir():
            continue

        for image_path in label_name.glob("*.png"):
            template = plt.imread(image_path)[:, :, :3].mean(axis=2)
            template = (template > 0).astype(np.uint8)

            labeled_image = label(template)
            regions = regionprops(labeled_image)

            if not regions:
                continue

            target_region = max(regions, key=lambda r: r.area)
            features.append(extract_features(target_region))
            labels_list.append(label_idx)
    return np.array(features), np.array(labels_list)


def train_knn(features, labels):
    knn = cv2.ml.KNearest_create()
    knn.train(features.astype(np.float32), cv2.ml.ROW_SAMPLE, labels.astype(np.float32).reshape(-1, 1))
    return knn


def apply_model_on_new_images(model, test_dir, labels):
    test_dir = Path(test_dir)

    for test_image in test_dir.glob("*.png"):
        print(f"{test_image.stem})", end=" ")

        template = plt.imread(test_image)[:, :, :3].mean(axis=2)
        template = (template > 0.1).astype(np.uint8)

        labeled_image = label(template)
        regions = regionprops(labeled_image)
        regions.sort(key=lambda r: r.centroid[1])

        last_x = None
        for region in regions:
            if region.area <= 250:
                continue

            features = extract_features(region).astype(np.float32).reshape(1, -1)
            _, results, _, _ = model.findNearest(features, k=2)

            bbox = region.bbox
            if last_x is not None and bbox[1] - last_x > 30:
                print(" ", end="")

            last_x = bbox[3]
            print(labels[int(results[0][0])][-1], end="")
        print()


if __name__ == "__main__":
    train_dir = "task/train/"
    test_dir = "task/"

    features, labels = load_training_data(train_dir)
    knn_model = train_knn(features, labels)

    labels_list = [d.name for d in Path(train_dir).iterdir() if d.is_dir()]
    apply_model_on_new_images(knn_model, test_dir, labels_list)
