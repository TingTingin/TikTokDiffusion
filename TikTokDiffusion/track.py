import argparse
import os
import random
import time

import orjson
import gc

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from PIL import Image, ImageDraw, ImageChops, ImageFilter
from enum import Enum
import sys
import numpy as np
from pathlib import Path
import torch
import glob
from dataclasses import dataclass, field, astuple
import string
from concurrent.futures import ThreadPoolExecutor
import facechecker
import webuiapi
import functools
import SSIM_PIL
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH

if str(ROOT / 'rife') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))

if r"TikTokDiffusion\rife" not in sys.path:
    sys.path.append(r"TikTokDiffusion\rife")

if r"rife" not in sys.path:
    sys.path.append(r"rife")

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import RIFE
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import (LOGGER, Profile, check_img_size, non_max_suppression, scale_boxes, check_requirements,
                                   print_args, scale_segments)
from yolov5.utils.torch_utils import select_device
from utils.segment.general import masks2segments, process_mask, process_mask_native
from trackers.multi_tracker_zoo import create_tracker

@torch.no_grad()
def run(
        output_path,
        default_save_folder,
        fps,
        sampler,
        frame_gen_all,
        frame_gen_auto,
        cfg,
        height,
        width,
        seed,
        denoising,
        pipe,
        frame_gen_even,
        restore_faces,
        server,
        audio_source,
        control_net_model,
        control_net_lowvram,
        control_net_processor,
        source: Path,
        yolo_weights=WEIGHTS / 'yolov5x-seg.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_crop=False,  # save cropped prediction boxes
        classes=0,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        retina_masks=True,
        debug=False,
        skip_if_exists="",
        check_faces=False,
        prompt_by_body=False,
        steps=20,
        save_format=".png",
        stable_diffusion_folder="",
        face_match_sensitivity=1,
        func_prompt_list=None,
        ignore_if_no_prompt=True,
        fallback=None,
        loopback=1,
        inject_last_image=dict,
        variation_scaling=0.0,
        seed_scaling=False,
        ssim_retry=1,
        ssim_threshold=0.87,
        max_person_size=150,
        skip_ratio=.80,
        max_frame_skip=2,
        use_control_net=False,
        negative_prompt="",
):

    print(inject_last_image)
    org_max_person_size = max_person_size
    max_person_size = org_max_person_size*org_max_person_size
    source = str(source)
    source_folder = os.path.dirname(source)
    dataset = []
    name_prompt = func_prompt_list[0] if len(func_prompt_list[0]) < 20 else func_prompt_list[0][:20]+"..."
    id_set = set()

    if check_faces: #this makes the prompts work from right to left which is how prompt to body works and is more intuitive
        func_prompt_list = list(reversed(func_prompt_list))

    if skip_if_exists:
        save_folder = os.path.join(output_path, skip_if_exists)
    else:
        save_folder = default_save_folder

    class SaveType(Enum):
        DEBUG_IMAGE = "debug"
        UNFOUND_IMAGE = "unfound"
        FOUND_IMAGE = "found"
        FACES = "faces"

    class PersonJob:
        class_save_format = save_format
        class_full_save_folder_path = stable_diffusion_folder

        def __init__(self, org_image_path):
            self.org_image_path = org_image_path
            self.org_image_array = None
            self.segment = None

            filename = os.path.splitext(os.path.basename(self.org_image_path))[0]
            safe_filename = name_cleaner(filename)

            self.save_name_no_ext = safe_filename
            self.full_save_path_with_ext = os.path.join(save_folder, self.save_name_no_ext+f"{save_format}")
            self.mask_with_alpha = None
            self.masked_image = None
            self.prompt = None
            self.generated_times = 0
            self.found_face = False
            self.label = None
            self.face_index = None
            self.face = None
            self.track_label = None

        def save_debug_image(self, folder: SaveType, count=0):
            debug_folder = os.path.join(save_folder, "debugging", folder.value)

            try:
                os.makedirs(debug_folder)
            except FileExistsError:
                pass

            save_full_file_name = os.path.join(debug_folder, self.save_name_no_ext + f"_{count}_{folder.value}.jpg")

            if folder == SaveType.FACES:
                if type(self.face) == Image:
                    self.face.save(save_full_file_name)
            elif folder == SaveType.FOUND_IMAGE:
                if self.label is not None:
                    dir_name, filename = os.path.split(save_full_file_name)
                    label_path = os.path.join(dir_name, self.prompt if self.prompt is not None else "No Prompt", filename)

                    try:
                        os.makedirs(os.path.dirname(label_path))
                    except FileExistsError:
                        pass

                    self.get_masked_image().convert("RGB").save(label_path)
                else:
                    self.get_masked_image().convert("RGB").save(save_full_file_name)
            else:
                self.get_masked_image().convert("RGB").save(save_full_file_name)

            print(f"saving {folder.value} image to {save_full_file_name}")

        def get_org_image(self):
            return Image.open(self.org_image_path)

        def get_masked_image(self, invert=True, get_hole=False):
            def get_mask_or_alpha(mode="RGB"):
                mask_image_alpha = Image.new(mode, self.get_org_image().size, color=(0, 0, 0, 0))
                draw = ImageDraw.Draw(mask_image_alpha, mode)
                draw.polygon(self.segment.segment, fill=(255, 255, 255))
                return mask_image_alpha

            if get_hole:
                return_mask = ImageChops.invert(get_mask_or_alpha("RGBA"))
            elif invert:
                mask_image_alpha = get_mask_or_alpha("RGBA")
                return_mask = self.get_org_image().convert("RGBA")
                return_mask.alpha_composite(ImageChops.invert(mask_image_alpha))
            else:
                mask_image = get_mask_or_alpha()
                return_mask = mask_image
            return return_mask

    @dataclass
    class PersonList:
        persons: list[PersonJob] = field(default_factory=list)
        found_persons: list[PersonJob] = field(default_factory=list)
        unfound_persons: list[PersonJob] = field(default_factory=list)
        group_by_image_path_list: list[list[PersonJob]] = field(default_factory=list)
        generated_images: list[PersonJob] = field(default_factory=list)

        def get_found_persons(self):
            self.found_persons = []
            self.unfound_persons = []

            for loop_person in self.persons:
                if loop_person.found_face:
                    self.found_persons.append(loop_person)
                else:
                    self.unfound_persons.append(loop_person)

        def save_faces(self, save_type: SaveType):
            pool = ThreadPoolExecutor()
            if save_type == SaveType.UNFOUND_IMAGE:
                list_of_persons = self.unfound_persons

            if save_type == SaveType.DEBUG_IMAGE:
                list_of_persons = self.persons

            if save_type == SaveType.FOUND_IMAGE:
                list_of_persons = self.found_persons

            workers = []
            for count, save_person in enumerate(list_of_persons):
                workers.append(pool.submit(save_person.save_debug_image, save_type, count))

            for worker in workers:
                worker.result()

        def get_prompts(self):
            def get_missing():
                if not check_faces and not prompt_by_body:
                    return
                if check_faces:
                    loop_times = range(2)
                else:
                    loop_times = range(1)

                for times in loop_times:
                    for person_count, person in enumerate(self.persons):
                        if check_faces:
                            checker = person.found_face is False and times != 0
                        else:
                            checker = person.track_label is None
                        person.track_label = person.segment.id

                        if checker:  # this means we found a segmentation map but there was no track_label so we dont know what prompt to give it
                            searching_for_box = True
                            current_frame = person.segment.frame_idx
                            before_count = person_count
                            after_count = person_count
                            run_times = 0

                            while searching_for_box:
                                run_times += 1
                                compare_list = []
                                after_count += 1

                                if current_frame != 0:
                                    before_count -= 1

                                    try:
                                        before = self.persons[before_count]
                                        if before.segment.frame_idx < current_frame:
                                            compare_list.append(before)
                                    except IndexError:
                                        pass

                                try:
                                    after = self.persons[after_count]
                                    if after.segment.frame_idx > current_frame:
                                        compare_list.append(after)
                                except IndexError:
                                    pass

                                if compare_list:
                                    for _, compare_person in enumerate(compare_list):
                                        if searching_for_box is False:
                                            continue

                                        if prompt_by_body:
                                            if compare_person.segment.id is None:
                                                continue

                                        bounding_boxes = zip(person.segment.bounding_box,
                                                             compare_person.segment.bounding_box)
                                        all_difference = 0
                                        for box in bounding_boxes:
                                            all_difference += abs(box[0] - box[1])
                                        if all_difference <= 100:
                                            if times == 0:
                                                person.track_label = compare_person.segment.id
                                                if person.track_label > len(func_prompt_list):
                                                    person.prompt = fallback
                                                else:
                                                    person.prompt = func_prompt_list[int(person.track_label) - 1]
                                                    print(person.prompt)
                                            else:
                                                if not compare_person.found_face:
                                                    continue

                                                person.label = compare_person.label
                                                if person.label > len(func_prompt_list):
                                                    person.prompt = fallback
                                                else:
                                                    if person.label+1 > len(func_prompt_list):
                                                        person.prompt = fallback
                                                    else:
                                                        person.prompt = func_prompt_list[int(person.label)]
                                                        person.found_face = True
                                                        print(person.prompt)

                                            searching_for_box = False

                                        elif person.track_label is not None:
                                            if person.track_label > len(func_prompt_list):
                                                person.prompt = fallback
                                            else:
                                                person.prompt = func_prompt_list[int(person.track_label) - 1]
                                                print(person.prompt)

                                if run_times >= 15:
                                    if not searching_for_box:
                                        continue

                                    print(f"searched {run_times} for match and didnt find")
                                    break

                            if person.prompt is None:
                                print(f"None person on frame {person.segment.frame_idx}")
                        elif prompt_by_body:
                            if person.track_label > len(func_prompt_list):
                                person.prompt = fallback
                            else:
                                person.prompt = func_prompt_list[int(person.track_label) - 1]
                                print(person.prompt)

            if check_faces:
                self.persons = facechecker.get_face_identities(self.persons, face_match_sensitivity)
                self.get_found_persons()

                for person in self.found_persons:
                    try:
                        person.prompt = func_prompt_list[int(person.label)]
                    except IndexError:
                        person.prompt = fallback

                get_missing()
                self.get_found_persons()
                if debug:
                    person_list.save_faces(SaveType.UNFOUND_IMAGE)
                    person_list.save_faces(SaveType.FOUND_IMAGE)
                    print("Finished Saving Debug Images")

            elif not prompt_by_body:
                for person in self.persons:
                    person.prompt = func_prompt_list[0]

                self.group_by_image_path()
                return

            get_missing()
            self.group_by_image_path()

        def group_by_image_path(self):
            person_images = {}
            for person in self.persons:
                found_path = person_images.get(person.org_image_path)
                if found_path is None:
                    person_images[person.org_image_path] = [person]
                else:
                    person_images[person.org_image_path].append(person)

            self.group_by_image_path_list = [value for value in person_images.values()]

        def add_to_generate_list(self, generated_class:PersonJob):
            generated_class.generated_times += 1
            self.generated_images.append(generated_class)

    @dataclass
    class SegmentCoord:
        segment: np.array
        frame_idx: int
        bounding_box: field(default_factory=list)
        path: str
        id = None
        dif = None

    @dataclass
    class SegmentListClass:
        segments: field(default_factory=list)

        def sort_segments(self):
            self.segments = sorted(self.segments, key=lambda x: x.path)

    person_list = PersonList()
    segment_list = SegmentListClass(list())
    host, port = str(server).split(":")
    api = webuiapi.WebUIApi(host=host, port=port)
    unfound_masks = []

    def name_cleaner(name_to_clean:str):
        valid_chars = f"-_.(), {string.ascii_letters}{string.digits}"
        safe_filename = "".join(letter for letter in name_to_clean if letter in valid_chars)
        return safe_filename

    def check_if_masks_exists_and_save_masks(list_of_files: list, current_image_segments=None):
        for _, image_path in enumerate(list_of_files):
            current_image_path = image_path[1] if type(image_path) == tuple else image_path
            not_found = True
            masks_folder = os.path.join(output_path, "masks")
            mask_filename = name_cleaner(current_image_path)
            mask_filename = os.path.join(masks_folder, mask_filename + f".json")

            if not os.path.exists(masks_folder):
                try:
                    os.makedirs(masks_folder)

                except FileExistsError:
                    pass

            else:
                if os.path.exists(mask_filename):
                    with open(mask_filename, "rb") as mask_file:
                        segments_tuple = orjson.loads(mask_file.read())

                    for segment_item in segments_tuple:
                        current_segment = SegmentCoord(segment=np.array(segment_item.get("segment"), dtype=np.float32),
                                                       frame_idx=segment_item.get("frame_idx"),
                                                       bounding_box=segment_item.get("bounding_box"),
                                                       path=segment_item.get("path"))
                        current_segment.id = segment_item.get("id")
                        current_segment.dif = segment_item.get("dif")
                        for value in astuple(current_segment):
                            if value is None and current_segment.frame_idx !=0:
                                break

                        segment_list.segments.append(current_segment)
                        id_set.add(current_segment.id)
                        not_found = False
            if current_image_segments is not None:
                with open(mask_filename, "wb") as mask_file:
                    mask_file.write(orjson.dumps(tuple(current_image_segments), option=orjson.OPT_SERIALIZE_NUMPY))

            if not not_found:
                print(f"found mask for {current_image_path} so no new mask will be generated for it\n")
            else:
                unfound_masks.append(current_image_path)

    check_mask_pool = ThreadPoolExecutor()
    check_mask_workers = []

    for check_image_path in enumerate(glob.glob(source+"\*")):
        check_mask_workers.append(check_mask_pool.submit(check_if_masks_exists_and_save_masks, [check_image_path]))

    for check_mask_worker in check_mask_workers:
        check_mask_worker.result()

    # Load model
    if unfound_masks:
        device = select_device(device)
        is_seg = '-seg' in str(yolo_weights)
        model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
        segment_list.sort_segments()
        dataset.files = sorted(unfound_masks)
        dataset.nf = len(unfound_masks)
        dataset.count = len(unfound_masks)

        # Create as many strong sort instances as there are video sources
        tracker_list = []
        for i in range(nr_sources):
            tracker = create_tracker(tracking_method, reid_weights, device, half)
            tracker_list.append(tracker, )
            if hasattr(tracker_list[i], 'model'):
                if hasattr(tracker_list[i].model, 'warmup'):
                    tracker_list[i].model.warmup()
        outputs = [None] * nr_sources

        seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
        curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
            with dt[0]:
                im = torch.from_numpy(im).to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:

                if is_seg:
                    pred, proto = model(im, augment=augment, visualize=visualize)[:2]
                else:
                    pred = model(im, augment=augment, visualize=visualize)

            # Apply NMS
            with dt[2]:
                if is_seg:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                else:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                curr_frames[i] = im0

                s += '%gx%g ' % im.shape[2:]  # print string
                imc = im0.copy() if save_crop else im0  # for save_crop

                if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                    if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                        tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

                if det is not None and len(det):
                    if is_seg:
                        # scale bbox first the crop masks
                        if retina_masks:
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                            masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                        else:
                            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                        # Segments
                        segments = [
                            scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=False)
                            for x in reversed(masks2segments(masks))]

                        current_image_segments = []
                        for j, segment in enumerate(segments):
                            segmentation_map = segment

                            # Find the minimum and maximum x and y values
                            min_x = min(x for x, y in segmentation_map)
                            max_x = max(x for x, y in segmentation_map)
                            min_y = min(y for x, y in segmentation_map)
                            max_y = max(y for x, y in segmentation_map)
                            area = (max_x - min_x) * (max_y - min_y)

                            if area <= max_person_size:
                                print(f"Skipping detected person on frame {frame_idx} as they are below the size limit of {org_max_person_size}")
                                continue

                            # Print the bounding box coordinates
                            current_segment = SegmentCoord(segment=segment, frame_idx=frame_idx, bounding_box=(min_x, min_y, max_x, max_y), path=path)
                            segment_list.segments.append(current_segment)
                            current_image_segments.append(current_segment)
                            if frame_idx == 0:
                                current_segment.id = None

                    else:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # pass detections to strongsort
                    with dt[3]:
                        outputs[i] = tracker_list[i].update(det.cpu(), im0)

                    # draw boxes for visualization

                    if len(outputs[i]) > 0:
                        for j, (output) in enumerate(outputs[i]):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            conf = output[6]

                            for a, segment in enumerate(current_image_segments):
                                left_dif = abs(segment.bounding_box[0] - bboxes[0])
                                top_dif = abs(segment.bounding_box[1] - bboxes[1])
                                bot_dif = abs(segment.bounding_box[2] - bboxes[2])
                                right_dif = abs(segment.bounding_box[3] - bboxes[3])
                                all_dif = left_dif + top_dif + bot_dif + right_dif
                                segment.dif = all_dif

                            smallest = None
                            for segment in current_image_segments:
                                if smallest is None:
                                    smallest = segment.dif
                                else:
                                    if segment.dif < smallest:
                                        smallest = segment.dif

                            for segment in current_image_segments:
                                if smallest == segment.dif:
                                    segment.id = id
                                    id_set.add(id)

                    if current_image_segments:
                        check_if_masks_exists_and_save_masks([path], current_image_segments)
                else:
                    pass

            try:
                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt]) * 1E3:.1f}ms")
            except AttributeError:
                print(LOGGER.info(f"{s} (no detections)"))

    try:
        del model
        del dt
        del pred
        torch.cuda.empty_cache()
        gc.collect()
    except UnboundLocalError:
        print("no model to delete")
        pass

    id_list = [id for id in id_set if id is not None]
    id_dict = {}

    for count, id_to_change in enumerate(id_list, 1):  #normalizing ids so that we can use them as indexes later
        if id_to_change is None:
            continue

        id_dict.update({id_to_change: count})

    for segment in segment_list.segments:
        person = PersonJob(segment.path)

        person.segment = segment

        person.segment.id = id_dict[person.segment.id] if segment.id is not None else None

        if debug or check_faces:
            person.masked_image = True

        person_list.persons.append(person)

    person_list.persons = sorted(person_list.persons, key=lambda x:x.org_image_path)
    segment_list.sort_segments()
    person_list.get_prompts() #assigns prompts to people and groups images by the path to the file
    ssim_persons = []
    last_generated_image_for_injection = None

    def save_generated_img(image: Image, full_path_to_save: str):
        save_img_dir = os.path.dirname(full_path_to_save)
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        try:
            try:
                image.save(full_path_to_save)
            except PermissionError:
                time.sleep(.1)
                save_generated_img(image, full_path_to_save)
                print(f"Error Saving: {full_path_to_save} retrying")
        except OSError:
            pass

        print(f"Saved image to {full_path_to_save}")
        ssim_persons.append({"image": image, "path": full_path_to_save})
        func_last_generated_image_for_injection = image
        return func_last_generated_image_for_injection

    def get_similar_frames(files_to_check):
        func_skip_set = set()
        for count, file in enumerate(files_to_check):
            if file in func_skip_set:
                continue

            skip_list = []
            frames_count = 0

            current_image = Image.open(file)
            for sub_count, sub_file in enumerate(files_to_check):
                if frames_count >= max_frame_skip:
                    skip_list.pop(-1)
                    for pathfile in skip_list:
                        func_skip_set.add(pathfile)
                    break

                if sub_count <= count:
                    continue

                next_image = Image.open(sub_file)
                sub_similar_ratio = SSIM_PIL.compare_ssim(current_image, next_image)

                if sub_similar_ratio*100 > skip_ratio:
                    frames_count += 1
                    skip_list.append(sub_file)
                else:
                    if skip_list:
                        skip_list.pop(-1)
                        for pathfile in skip_list:
                            func_skip_set.add(pathfile)
                    break
        return func_skip_set

    def save_video(path_to_images, framerate, filename, generated_images_folder=""):

        filename = name_cleaner(filename)
        if not framerate:
            framerate = 30

        non_generated_images = glob.glob(path_to_images+"\*.png")
        generated_images = glob.glob(path_to_images+"\generated_imgs\*.png")

        if generated_images_folder:
            generated_images_folder = glob.glob(generated_images_folder+"\*.png")

        if not generated_images_folder:
            images = non_generated_images
        else:
            images = sorted(non_generated_images+generated_images_folder, key=lambda x: os.path.basename(x))

        sequence = ImageSequenceClip(images, framerate)
        sequence_save_folder = os.path.join(output_path, "videos")

        if not os.path.exists(sequence_save_folder):
            os.makedirs(sequence_save_folder)

        save_filename = os.path.join(sequence_save_folder, filename+f" {random.random()}.mp4")

        sequence.write_videofile(save_filename, framerate, audio=audio_source)

    def skip_list_split(list_to_split, sections=5):
        list_length = len(list_to_split)
        section_size = list_length // sections
        lists = []
        loop_range = range(sections)
        loop_range_size = len(loop_range)
        for section in loop_range:
            section += 1
            slice_amt = section * section_size

            if section == loop_range_size:
                lists.append(list_to_split[slice_amt - section_size:])
            else:
                lists.append(list_to_split[slice_amt - section_size:slice_amt + max_frame_skip + 1])
        return tuple(lists)

    skip_set = set()
    files_in_source_dir = list(glob.glob(source + "\*.png"))

    if frame_gen_auto:
        print(f"Checking for frames to skip using a ratio of {skip_ratio}")
        skip_workers = []
        skip_pool = ThreadPoolExecutor()
        skip_lists = skip_list_split(files_in_source_dir, 6)

        for skip_list in skip_lists:
            res = skip_pool.submit(get_similar_frames, skip_list)
            skip_workers.append(res)

        for skip_worker in skip_workers:
            skip_set.update(skip_worker.result())

        print(f"Found {len(skip_set)} similar image(s) to skip")

    total_jobs = 0
    for count_job in person_list.group_by_image_path_list:
        total_jobs += len(count_job)
    total_jobs = total_jobs-len(skip_set)
    org_job_length = total_jobs-len(skip_set)
    done_jobs = 0
    generated_list = {}
    last_time = 0
    for personjob_list_count, personjob_list in enumerate(person_list.group_by_image_path_list):

        if frame_gen_all or frame_gen_even:
            if personjob_list_count % 2 != 0:
                continue

        ssim_loop = range(ssim_retry) if personjob_list_count != 0 else range(1)
        best_ssim = 0
        variation = 0
        skip_this_frame = False
        #hack need to change later

        if skip_set:
            for set_personjob in personjob_list:
                if set_personjob.org_image_path in skip_set:
                    skip_this_frame = True
                    break

        if skip_if_exists:
            for check_personjob in personjob_list:
                if os.path.exists(check_personjob.full_save_path_with_ext):
                    print(f"file: {check_personjob.full_save_path_with_ext} exists so it will be skipped. And no new image will be generated")
                    skip_this_frame = True
                    break

        if skip_this_frame:
            done_jobs += len(personjob_list)
            remaining_jobs = total_jobs-done_jobs
            print(f"{remaining_jobs} jobs remaining")

            for personjob in personjob_list:
                if generated_list.get(personjob.prompt) is None:
                    generated_list[personjob.prompt] = 1
                else:
                    generated_list[personjob.prompt] += 1
            continue

        for ssim_amount in ssim_loop:
            ssim_amount += 1
            doing_ssim = ssim_amount > 1

            if doing_ssim and variation_scaling != 0:
                variation = variation_scaling

            if doing_ssim and len(ssim_persons) > 1:
                main_image = ssim_persons[0]["image"]
                last_generated_image_to_compare = ssim_persons[-1]["image"]
                ssim_difference = SSIM_PIL.compare_ssim(main_image, last_generated_image_to_compare)
                print(f"Compared Image 1 to Image {len(ssim_persons)} and they were {round(ssim_difference, 3)*100}% similar")

                if ssim_difference >= ssim_threshold/100:
                    print(f"This is above the similarity threhsold of {ssim_threshold}% that was set so this Image has been selected")
                    last_generated_image_for_injection = save_generated_img(last_generated_image_to_compare, ssim_persons[-1]["path"])

                    done_jobs += 1
                    remaining_jobs = total_jobs - done_jobs
                    print(f"{remaining_jobs} jobs remaining")

                    ssim_persons = []
                    break
                else:
                    print(f"This is below the similarity threhsold of {ssim_threshold} that was set so will keep searching {ssim_amount}/{ssim_retry} times")
                    done_jobs -= 1
                    if ssim_difference > best_ssim:
                        best_ssim = ssim_difference
                        best_ssim_image = last_generated_image_to_compare
                        best_ssim_image_path = ssim_persons[-1]["path"]

            if ssim_amount == ssim_retry and doing_ssim:
                print(f"Reached search limit {ssim_amount}/{ssim_retry} limit choosing most similar image in batch which was {round(best_ssim, 3)*100}% similar")
                last_generated_image_for_injection = save_generated_img(best_ssim_image, best_ssim_image_path)

                done_jobs += 1
                remaining_jobs = total_jobs - done_jobs
                print(f"{remaining_jobs} jobs remaining")

                ssim_persons = []
                break

            completed_images = []

            for person_count, personjob in enumerate(personjob_list, 1):
                if personjob.prompt is None:
                    if ignore_if_no_prompt:
                        continue
                    else:
                        completed_images = []
                        for sub_personjob in personjob_list:
                            sub_personjob.generated_times = 0
                        break

                for loop_times in range(loopback):
                    loop_times = loop_times
                    loop_times += 1
                    print_prompt = personjob.prompt if len(personjob.prompt) < 60 else personjob.prompt[:60]+"..."

                    if loop_times > 1:
                        print(f"generating prompt: '{print_prompt}' for person {person_count} again since loopback is {loopback} current: {loop_times}/{loopback}")

                    if loopback > 1 and loop_times > 1 and person_list.generated_images:
                        if not use_control_net:
                            gen_image = image.images[-1]
                        else:
                            gen_image = image.images[0]

                    elif inject_last_image and last_generated_image_for_injection is not None:
                        last_generated_image = last_generated_image_for_injection
                        last_generated_image = last_generated_image.filter(ImageFilter.GaussianBlur(inject_last_image["Blur"]))
                        mask = ImageChops.invert(personjob.get_masked_image(get_hole=True))
                        last_generated_image = Image.composite(last_generated_image, mask, mask).convert("RGBA")
                        last_generated_image_array = np.array(last_generated_image)
                        opaque = last_generated_image_array[:, :, 3] == 255
                        opacity = inject_last_image["Opacity"]
                        last_generated_image_array[opaque, 3] = opacity * 255
                        last_generated_image = Image.fromarray(last_generated_image_array).convert("RGBA")
                        blended_image = Image.alpha_composite(personjob.get_org_image().convert("RGBA"), last_generated_image)
                        gen_image = blended_image
                        print(f"Injecting previous image into generation at blur {inject_last_image['Blur']} and opacity {inject_last_image['Opacity']}")

                    else:
                        gen_image = personjob.get_org_image()

                    if loop_times <= 1:
                        print(f"generating prompt: '{print_prompt}' for person {person_count}")
                    api_ssim_amount = ssim_amount if seed_scaling else 0

                    if pipe.poll():
                        instruction = pipe.recv()
                        if instruction == "STOP":
                            return False

                    prompt = personjob.prompt
                    gen_times = generated_list.get(prompt)

                    transition_frames = 30
                    mid_transition_frames = 15

                    if loop_times <= 1:
                        print(f"generating prompt: '{print_prompt}' for person {person_count}")

                    if last_time != 0:
                        cur_time = time.perf_counter()
                        runtime = cur_time-last_time
                        print(runtime)
                        total_time = round(runtime*total_jobs)
                        print("")
                        print(f"estimated time remaining {total_time} seconds")
                        print("")

                    last_time = time.perf_counter()
                    image = api.img2img(images=[gen_image],
                                        prompt=prompt,
                                        mask_image=personjob.get_masked_image(False),
                                        inpainting_fill=1,
                                        cfg_scale=cfg,
                                        denoising_strength=denoising,
                                        inpaint_full_res=True,
                                        steps=steps,
                                        sampler_index=sampler,
                                        restore_faces=restore_faces,
                                        negative_prompt=negative_prompt,
                                        height=height,
                                        width=width,
                                        seed=seed+loop_times+api_ssim_amount,
                                        subseed=ssim_amount,
                                        subseed_strength=variation,
                                        use_control_net=use_control_net,
                                        controlnet_model=control_net_model,
                                        controlnet_module=control_net_processor,
                                        controlnet_lowvram=control_net_lowvram
                                        )

                    person_list.add_to_generate_list(personjob)

                    if loop_times == loopback:

                        done_jobs += 1
                        remaining_jobs = total_jobs - done_jobs
                        print(f"{remaining_jobs} jobs remaining")

                        if use_control_net:
                            image.images = [image.images[0]]

                        completed_images.append([image.images[-1], personjob.get_masked_image(get_hole=True)])

            if len(completed_images) == 1:
                last_generated_image_for_injection = save_generated_img(completed_images[0][0], personjob_list[0].full_save_path_with_ext)

            elif len(completed_images) > 1:
                final_image = functools.reduce(lambda x, y: Image.composite(x[0] if type(x) is list else x, y[0], y[1]), completed_images)
                last_generated_image_for_injection = save_generated_img(final_image, personjob_list[0].full_save_path_with_ext)
            print("")

    generated_images_save_folder = os.path.join(save_folder, "generated_imgs")
    if frame_gen_auto or frame_gen_even:

        if frame_gen_auto:
            print(f"Generating {len(skip_set)} skipped frames")

        RIFE.frame_generator(path_of_images=save_folder, save_folder=generated_images_save_folder)
        save_video(path_to_images=save_folder, filename=name_prompt, framerate=fps, generated_images_folder=generated_images_save_folder)

    elif frame_gen_all:
        RIFE.frame_generator(path_of_images=save_folder, save_folder=generated_images_save_folder)
        RIFE.frame_generator(path_of_images=generated_images_save_folder, save_folder=generated_images_save_folder)
        save_video(path_to_images=generated_images_save_folder, filename=name_prompt, framerate=fps)

    else:
        save_video(path_to_images=save_folder, filename=name_prompt, framerate=fps)

    print(f"Saving video to {os.path.join(output_path, 'videos')}")
    return True
