import glob
import os

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from .backbone import get_model


@torch.no_grad()
def inference(name, weight, src_norm_dir):
    face_model = get_model(name)([112, 112])
    face_model.load_state_dict(torch.load(weight, map_location="cpu"))
    face_model = face_model.to("cpu")
    face_model.eval()

    id2src_norm = {}
    for src_id in sorted(list(os.listdir(src_norm_dir))):
        id2src_norm[src_id] = sorted(list(glob.glob(f"{os.path.join(src_norm_dir, src_id)}/*")))

    total_sims = []
    for id_name in tqdm(id2src_norm):
        src_face_embeddings = []
        for src_img_path in id2src_norm[id_name]:
            src_img = cv2.imread(src_img_path)
            src_img = cv2.resize(src_img, (112, 112))
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            src_img = np.transpose(src_img, (2, 0, 1))
            src_img = torch.from_numpy(src_img).unsqueeze(0).float()
            src_img.div_(255).sub_(0.5).div_(0.5)
            embedding = face_model(src_img).detach().cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            src_face_embeddings.append(embedding)  # 512

        num = len(src_face_embeddings)
        src_face_embeddings = np.stack(src_face_embeddings)  # n, 512
        sim = src_face_embeddings @ src_face_embeddings.T  # n, n
        mean_sim = (np.sum(sim) - num * 1.0) / ((num - 1) * num)
        print(f"{id_name}: {mean_sim}")
        total_sims.append(mean_sim)

    return np.mean(total_sims)


if __name__ == "__main__":
    name = 'IR_101'
    weight = "models/glint360k_curricular_face_r101_backbone.bin"
    src_norm_dir = "/disk1/hejunjie.hjj/data/normface-AFD-id-20"
    mean_sim = inference(name, weight, src_norm_dir)
    print(f"total: {mean_sim:.4f}")  # total: 0.6299
