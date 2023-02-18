import time
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from dface import FaceNet, MTCNN
device = 'cpu'


def get_boundingbox(box, w, h, scale=1.2):
    x1, y1, x2, y2 = box
    size = int(max(x2-x1, y2-y1) * scale)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    if size > w or size > h:
        size = int(max(x2-x1, y2-y1))
    x1 = max(int(center_x - size // 2), 0)
    y1 = max(int(center_y - size // 2), 0)
    size = min(w - x1, size)
    size = min(h - y1, size)
    return x1, y1, size


def get_face_identities(list_of_images: list, sensitivity:int):
    frames_path_num = len(list_of_images)
    print("Loading face checking model (MTCNN) since check faces is on.")
    mtcnn = MTCNN("cpu")

    frames = []
    faces = []
    found_frames = []
    face_index = 0
    print("Detecting faces.")
    for count, frame in enumerate(list_of_images, 1):
        frames.append(np.array(frame.get_masked_image().convert("RGB")))
        before = time.perf_counter()
        batch_size = 1000
        if count % batch_size == 0 or frames_path_num-count <= 0:
            result = mtcnn.detect(frames)
            print(f"Finished Face Search For {count}/{frames_path_num} Images")
            print(f"This Batch Took {time.perf_counter()-before} seconds")
            for i, res in enumerate(result):
                frame.face_index = face_index
                face_index += 1
                if res is None:
                    continue
                # extract faces
                boxes, probs, lands = res

                if len(boxes) > 2:
                    all_faces = zip(boxes, probs, lands)
                    largest_val = 0
                    largest_face = None
                    for sub_face in all_faces:
                        if sub_face[1] > largest_val:
                            largest_val = sub_face[1]
                            largest_face = sub_face
                else:
                    largest_face = next(zip(boxes, probs, lands))

                if largest_face[1] > 0.75:
                    h, w = frames[i].shape[:2]
                    x1, y1, size = get_boundingbox(largest_face[0], w, h)
                    face = frames[i][y1:y1+size, x1:x1+size]
                    found_frames.append({"found_frame": frame, "found_index": face_index - 1, "label": None})
                    faces.append(face)
            frames = []

    print("Loading face checking models (FaceNet).")
    facenet = FaceNet("cuda" if torch.cuda.is_available() else "cpu")

    embeds = facenet.embedding(faces)
    DBSCAN_sensitivity = sensitivity/10

    print(f"Grouping faces by identity current sensitvity: is {sensitivity}")
    dbscan = DBSCAN(eps=DBSCAN_sensitivity, metric='cosine', min_samples=5)
    labels = dbscan.fit_predict(embeds)

    for i in range(len(labels)):
        label = labels[i]

        if label < 0:
            continue

        list_of_images[found_frames[i]["found_index"]].label = label
        list_of_images[found_frames[i]["found_index"]].found_face = True

    return list_of_images
