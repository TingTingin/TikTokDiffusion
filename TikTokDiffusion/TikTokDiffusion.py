import shutil
import time
import glob
import os.path
import subprocess
import importlib
import sys
import re

reqs = {'gitpython': ['pip show gitpython', "WARNING"],
        'ipython': ['pip show ipython', "WARNING"],
        'matplotlib': "matplotlib>=3.2.2",
        'numpy': "numpy<=1.23.5",
        'cv2': "opencv-python>=4.1.1'",
        'orjson': "orjson",
        'PIL': "Pillow>=7.1.2",
        'psutil': "psutil",
        'yaml': "PyYAML>=5.3.1",
        'requests': "requests>=2.23.0",
        'scipy': "scipy>=1.4.1",
        'thop': "thop>=0.1.1",
        'SSIM_PIL': "SSIM-PIL",
        'moviepy': "moviepy",
        'sklearn': "scikit-learn",
        'dface': "dface",
        'tqdm': "tqdm>=4.64.0",
        'flet': "flet",
        'static_ffmpeg': "static-ffmpeg",
        'ffmpeg': "ffmpeg-python",
        'tensorboard': "tensorboard",
        'pandas': "pandas",
        'seaborn': "seaborn>=0.11.0",
        'easydict': "easydict>=0.11.0",
        'gdown': "gdown",
        'lap': r"TikTokDiffusion\Assets\whls\lap-0.4.0-cp310-cp310-win_amd64.whl",
        'cython_bbox': r"TikTokDiffusion\Assets\whls\cython_bbox-0.1.3-cp310-cp310-win_amd64.whl",
        'filterpy': "filterpy",
        'pyopencl': 'pyopencl',
        'yt_dlp': "yt-dlp"}


def check_requirements():

    print("Checking TikTokDiffusion...")
    python = sys.executable

    for search_name, req_command in reqs.items():
        package_search = importlib.util.find_spec(re.sub("[><]=.+", "", search_name))
        print(f"Checking TikTokDiffusion Requirement: {search_name}")

        present = package_search is not None
        if not present:
            if type(req_command) == list:
                try:
                    pipoutput = subprocess.run(fr'{req_command[0]}', capture_output=True, text=True)
                except Exception as e:
                    print(e)

                if not pipoutput.stdout.find("WARNING") == -1:
                    print(f"Installing TikTokDiffusion Requirement: {search_name}")
                    subprocess.run(fr'"{python}" -m pip install {req_command}')
            else:
                print(f"Installing TikTokDiffusion Requirement: {req_command}")
                try:
                    subprocess.run(fr'"{python}" -m pip install {req_command}')
                except Exception as e:
                    print(e)


check_requirements()

import PIL.Image
import flet as ft
import requests
import track
from pathlib import Path
import pprint
import base64
import webuiapi
from io import BytesIO
from statistics import mean
from collections import deque
import threading
import ffmpeg
import cv2
import string
import re
from multiprocessing import Pipe
import static_ffmpeg
import yt_dlp
from yt_dlp import YoutubeDL

static_ffmpeg.add_paths()
shutil.copy("webui-user.bat", "TikTok-Diffusion-webui-user.bat")
with open("TikTok-Diffusion-webui-user.bat", "r") as bat_file:
    contents = bat_file.read()
    replacement = r'\1\2 --api'
    output_text = re.sub("(set COMMANDLINE_ARGS=.*)((?<!--api).*)$", replacement, contents, flags=re.MULTILINE)

    with open("TikTok-Diffusion-webui-user.bat", "w") as tiktok_bat:
        tiktok_bat.write(output_text)

command = 'start /wait cmd /c TikTok-Diffusion-webui-user.bat'
process = subprocess.Popen(command, shell=True)

print("Launching Stable Diffusion With TikTokDiffusion")
stable_diffusion_folder = str(Path(__file__).parent.parent)
stable_diffusion_save_subfolder = r"outputs\Tiktok Diffusion"
output_path = os.path.join(stable_diffusion_folder, stable_diffusion_save_subfolder)
images_from_video_folder = os.path.join(output_path, "Images From Videos")

url_source = False
generating = False
video_fps = None
video_source = False
ui, stable_diffusion_conn = Pipe()


def get_default_save_folder():
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files_in_output_path = os.listdir(output_path)
    folder_prefix = "video_"

    for i in range(1000):
        i += 1
        folder_name = folder_prefix + str(len(files_in_output_path) + i)
        save_folder = os.path.join(output_path, folder_name)
        if not os.path.exists(save_folder):
            break

    return save_folder


default_save_folder = get_default_save_folder()


def frame_to_time(fps):
    time_skip = round(1/fps, 10)
    return time_skip


show_rate = 30
frame_rate = frame_to_time(show_rate)


class Slider:
    def __init__(self , page: ft.Page, name, min_val, max_val, divisions, data_type, starting_val, small_num=False):
        self.data_type = data_type
        self.small_num = small_num
        self.text = ft.Text(value=name, font_family="Main", size=20)
        self.text_container = ft.Container(content=self.text, padding=ft.padding.only(18))
        self.slider = ft.Slider(min=min_val, max=max_val, value=starting_val, divisions=divisions, on_change=self.update_text, label="{value}", expand=True)
        self.small_num = small_num
        if small_num:
            self.text.size = 20
            self.slider_amount = ft.TextField(value=self.data_type(self.slider.value),
                                              text_style=ft.TextStyle(size=15, font_family="Main"), width=150,
                                              height=25, on_change=self.update_slider)
        else:
            self.slider_amount = ft.TextField(value=self.data_type(self.slider.value),
                                              text_style=ft.TextStyle(size=25, font_family="Main"), width=150,
                                              height=70, on_change=self.update_slider)

        self.slider_row = ft.Row([self.slider, self.slider_amount])
        self.page = page

    def slider_creator(self):
        if self.small_num:
            slider_amount_container = ft.Container(self.slider_amount, padding=ft.padding.only(18))
            layout = ft.Column([self.text_container, self.slider, slider_amount_container], spacing=0)
        else:
            layout = ft.Column([self.text_container, self.slider_row], spacing=0)

        main_container = ft.Container(content=layout)
        if self.small_num:
            main_container = ft.Container(content=layout, expand=True)
        return main_container

    def update_text(self, label):
        self.slider_amount.value = round(self.data_type(self.slider.value), 3)

        if self.slider.value == 0:
            self.error()
        else:
            self.slider_amount.error_text = None

        self.page.update()

    def update_slider(self, label):
        text_value = 0
        try:
            text_value = self.data_type(self.slider_amount.value)
            self.slider.value = self.data_type(text_value)
            self.slider_amount.error_text = None

        except ValueError:
            self.error()

        if text_value == 0:
            self.error()
        else:
            self.slider_amount.error_text = None

        self.page.update()

    def error(self):
        self.slider_amount.error_text = "Enter A Valid Number"
        self.page.update()


class Button:
    def __init__(self, text, on, page, on_click_extra_on=None, on_click_extra_off=None):
        self.on_text = "ON"
        self.off_text = "OFF"
        self.button = ft.ElevatedButton(on_click=self.toggle, expand=True)
        self.page = page
        self.on_click_extra_on = on_click_extra_on
        self.on_click_extra_off = on_click_extra_off

        if on:
            self.turn_on()
        else:
            self.turn_off()

        self.text = ft.Text(value=text, font_family="Main", size=20)
        self.button_container = ft.Container(content=self.button, width=300, expand=True)

    def button_creator(self):
        layout = ft.Column([self.text, self.button_container], height=100)
        main_container = ft.Container(content=layout, expand=True, animate_scale=300)
        return main_container

    def on_press(self, event):
        self.button.style = ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10), color="Red")

    def turn_on(self):
        if self.on_click_extra_on is not None:
            self.on_click_extra_on()

        self.button.text = "ON"
        self.button.style = ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=5), bgcolor=ft.colors.GREEN, color="white")
        self.page.update()

    def turn_off(self):

        if self.on_click_extra_off is not None:
            self.on_click_extra_off()

        self.button.text = "OFF"
        self.button.style = ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=5), bgcolor=ft.colors.RED,color="white")
        self.page.update()

    def toggle(self, event):

        if self.button.text == "ON":
            self.turn_off()
        else:
            self.turn_on()


class EnterNumber:
    def __init__(self, name, number, data_type, page, border=ft.InputBorder.OUTLINE, fps=False):
        self.data_type = data_type
        self.page = page
        self.border = border
        self.fps = fps
        self.textfield = ft.TextField(value=number, label=name, label_style=(ft.TextStyle(font_family="Main", size=30, color=ft.colors.BLACK)), on_change=self.update_text, border=self.border)

    def enter_number_creator(self):
        main_container = ft.Container(self.textfield, expand=True)
        return main_container

    def update_text(self, label):
        try:
            self.data_type(self.textfield.value)
            self.textfield.error_text = None
            self.page.update()

        except ValueError:
            self.textfield.error_text = "Enter A Valid Number"

        if self.fps and not self.textfield.error_text:
            self.update_framerate()

        self.page.update()

    def update_framerate(self):
        global show_rate

        new_show_rate = round(float(self.textfield.value))
        show_rate = new_show_rate
        self.page.update()


class DropDown:
    def __init__(self, options, name, default_val, on_change=None):
        option_list = []

        for option in options:
            option = ft.dropdown.Option(option)
            option_list.append(option)

        self.dropdown = ft.Dropdown(options=option_list, label=name, value=default_val,label_style=(ft.TextStyle(font_family="Main", size=25, color=ft.colors.BLACK)))

        if on_change:
            self.dropdown.on_change = on_change

    def dropdown_creator(self):
        main_container = ft.Container(self.dropdown, expand=True, alignment=ft.alignment.top_center)
        return main_container


class TextFieldButton:
    def __init__(self, button_text=None, field_name=None, hint_text="Enter Prompt Here", no_button=False, border=ft.InputBorder.OUTLINE, default_field_val="", on_click=None, on_submit=""):
        self.button_text = button_text
        self.no_button = no_button
        self.on_submit = on_submit
        self.textfield = ft.TextField(value=default_field_val, label=field_name, label_style=(ft.TextStyle(font_family="Main", size=30, color=ft.colors.BLACK)), height=70, expand=1, autofocus=True, hint_text=hint_text,
                                      helper_style=ft.TextStyle(size=12, color=ft.colors.GREEN, weight=ft.FontWeight.BOLD), border=border)

        if self.on_submit:
            self.textfield.on_submit = on_submit
        self.button = ft.ElevatedButton(text=self.button_text, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=5), bgcolor=ft.colors.GREEN, color="white"), height=65)
        if on_click is not None:
            self.button.on_click = on_click
        self.button_container = ft.Container(content=self.button, padding=ft.padding.only(bottom=6))

    def textfield_button_creator(self):
        if self.no_button:
            layout = ft.Row([self.textfield])
        else:
            layout = ft.Row([self.textfield, self.button_container])

        main_container = ft.Container(layout)
        return main_container


class ImageContainer:
    def __init__(self, page=None, on_click=None):
        self.image = ft.Image(src="Icons/add image.png", gapless_playback=True, expand=True, fit=ft.ImageFit.FIT_HEIGHT)
        self.progress_bar = ft.ProgressBar(bar_height=0, bgcolor=ft.colors.TRANSPARENT)
        self.container = ft.Container(content=self.image, bgcolor=ft.colors.BLACK12, alignment=ft.alignment.center, expand=True, border_radius=5, on_click=on_click)

        self.fps_control = EnterNumber("Fps", show_rate, int, page, fps=True)
        self.fps = self.fps_control.enter_number_creator()
        self.fps_container = ft.Container(self.fps, alignment=ft.alignment.bottom_right, padding=10, width=200)

        self.source_control = TextFieldButton(no_button=True, field_name="Source Location", hint_text="No Source Selected", default_field_val="No Source Selected", border=ft.InputBorder.NONE)
        self.source_control.textfield.disabled = True
        self.source = self.source_control.textfield_button_creator()

        self.source_container = ft.Container(self.source, alignment=ft.alignment.top_right, padding=ft.padding.only(10))
        self.stack_top = ft.Column([self.source_container, self.fps_container], expand=True)

        self.page = page

    def image_container_creator(self):
        stack = ft.Stack([self.container, self.stack_top, self.progress_bar], expand=True)
        return stack


class Trio:
    def __init__(self, dropdown_name, options, page, slider_one_name, slider_two_name, slider_one_min, slider_one_max,
                 slider_two_min, slider_two_max, slider_one_divisions, slider_two_divisions, slider_one_datatype, slider_two_datatype, slider_one_starting_val, slider_two_starting_val, default_dropdown_val,
                 slider_one_enabled, slider_two_enabled, extra_control=None):

        self.slider_one_enabled = slider_one_enabled
        self.slider_two_enabled = slider_two_enabled
        self.page = page
        self.extra_control = extra_control
        if self.extra_control is not None:
            self.extra_control.slider.disabled = True

        self.dropdown_control = DropDown(name=dropdown_name, options=options, default_val=default_dropdown_val, on_change=self.enabler)
        self.dropdown = self.dropdown_control.dropdown_creator()
        self.dropdown.expand = 2

        self.slider_one_control = Slider(page=page, name=slider_one_name, min_val=slider_one_min,
                                         max_val=slider_one_max, divisions=slider_one_divisions,
                                         starting_val=slider_one_starting_val, data_type=slider_one_datatype,
                                         small_num=True)

        self.slider_one = self.slider_one_control.slider_creator()
        self.slider_one_control.slider.disabled = True
        self.slider_one_control.slider_amount.disabled = True

        self.slider_two_control = Slider(page=page, name=slider_two_name, min_val=slider_two_min, max_val=slider_two_max,
                                 divisions=slider_two_divisions, starting_val=slider_two_starting_val,
                                 data_type=slider_two_datatype, small_num=True)
        self.slider_two = self.slider_two_control.slider_creator()
        self.slider_two_control.slider.disabled = True
        self.slider_two_control.slider_amount.disabled = True

    def trio_creator(self):
        layout = ft.Row([self.dropdown, self.slider_one, self.slider_two], spacing=0)

        if self.extra_control is not None:
            layout.controls.append(self.extra_control.slider_creator())

        main_container = ft.Container(layout, border_radius=5, margin=5, padding=7)
        return main_container

    def enabler(self, event):
        for enable_condition in self.slider_one_enabled:

            if self.extra_control is not None:
                if event.data.lower() == "OFF".lower():
                    self.extra_control.slider.disabled = True
                    self.extra_control.slider_amount.disabled = True
                else:
                    self.extra_control.slider.disabled = False
                    self.extra_control.slider_amount.disabled = False
                    self.page.update()

            if enable_condition.lower() == str(event.data).lower():
                self.slider_one_control.slider.disabled = False
                self.slider_one_control.slider_amount.disabled = False
                self.page.update()
                break
            else:
                self.slider_one_control.slider.disabled = True
                self.slider_one_control.slider_amount.disabled = True
                self.page.update()

        for enable_condition in self.slider_two_enabled:
            if enable_condition.lower() == str(event.data).lower():
                self.slider_two_control.slider.disabled = False
                self.slider_two_control.slider_amount.disabled = False
                self.page.update()
                break
            else:
                self.slider_two_control.slider.disabled = True
                self.slider_two_control.slider_amount.disabled = True
                self.page.update()


class SourceButton:
    def __init__(self, name):
        self.button = ft.ElevatedButton(text=name, style=ft.ButtonStyle(bgcolor=ft.colors.BLUE, color=ft.colors.WHITE, shape=ft.RoundedRectangleBorder(radius=5)), expand=True)

    def source_button_creator(self):
        return self.button


class PromptButton:
    def __init__(self, name):
        self.button = ft.ElevatedButton(text=name, style=ft.ButtonStyle(bgcolor=ft.colors.BLUE, color=ft.colors.WHITE, shape=ft.RoundedRectangleBorder(radius=5)))

    def prompt_button_creator(self):
        main_container = ft.Container(content=self.button, padding=7)
        return main_container


def get_base64(image_path):
    buffered = BytesIO()
    success = False

    while not success:
        try:
            image = PIL.Image.open(image_path)
            image.save(buffered, format="JPEG")
            success = True

        except OSError:
            time.sleep(.1)
            pass

    my_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    if my_string is None:
        print("error")
    return my_string


def main(page: ft.Page):
    page.fonts = {"Main": "Fonts/CodeNext-Trial-Bold.otf"}
    progress_bar = ft.ProgressBar(height=5)
    server = "127.0.0.1:7860"
    server_link = ft.TextField(label="Server", label_style=(ft.TextStyle(font_family="Main", size=30, color=ft.colors.BLACK)), height=70, hint_text="Server", value=server)
    progress_text = ft.Text("Waiting For Stable Diffusion", font_family="Main", animate_scale=1000)
    progress = ft.Column([server_link, progress_bar], height=75)
    stable_waiter = ft.AlertDialog(title=progress_text, shape=ft.RoundedRectangleBorder(radius=5), content=progress, modal=True)
    stable_waiter.open = True
    page.add(stable_waiter)

    for i in range(100):
        time.sleep(1)
        try:
            response = requests.get(r"http://"+server_link.value)
            if response.status_code == 200:
                print(f"Successfully connected to {server_link.value}")
                stable_waiter.open = False
                break
        except Exception as e:
            pass

    server = server_link.value
    time.sleep(3)
    image_types = (".png", ".jpg")
    video_types = (".mp4", ".webm")
    image_error_text = "Found No Images In This Folder"
    image_source = None
    cfg_slider_control = Slider(name="CFG", min_val=0, max_val=50, page=page, divisions=50, data_type=int, starting_val=7)
    cfg_slider = cfg_slider_control.slider_creator()

    width_slider_control = Slider(name="Width", min_val=0, max_val=2048, page=page, divisions=32, data_type=int, starting_val=512)
    width_slider = width_slider_control.slider_creator()

    height_slider_control = Slider(name="Height", min_val=0, max_val=2048, page=page, divisions=32, data_type=int, starting_val=512)
    height_slider = height_slider_control.slider_creator()

    steps_slider_control = Slider(name="Steps", min_val=0, max_val=200, page=page, divisions=200, data_type=int, starting_val=20)
    steps_slider = steps_slider_control.slider_creator()

    denoising_slider_control = Slider(name="Denoising", min_val=0, max_val=1, page=page, divisions=100, data_type=float,starting_val = 0.05)
    denoising_slider = denoising_slider_control.slider_creator()

    face_match_slider_control = Slider(name="Face Match", min_val=0, max_val=1, page=page, divisions=100, data_type=float, starting_val=0.1, small_num=True)
    face_match_slider = face_match_slider_control.slider_creator()

    loopback_slider_control = Slider(name="LoopBack", min_val=1, max_val=20, page=page, divisions=19, data_type=int, starting_val=1, small_num=True)
    loopback_slider = loopback_slider_control.slider_creator()
    option_list = []
    sampler_options = webuiapi.WebUIApi().get_samplers()

    for option in sampler_options:
        sampler = option.get("name")
        if sampler is not None:
            option_list.append(sampler)

    sampler_dropdown_control = DropDown(name="Sampler", options=option_list, default_val="Euler")
    sampler_dropdown = sampler_dropdown_control.dropdown_creator()

    restore_faces_button_control = Button("Restore Faces", True, page)
    restore_faces_button = restore_faces_button_control.button_creator()

    skip_if_exists_button_control = Button("Skip If Exists", True, page)
    skip_if_exists_button = skip_if_exists_button_control.button_creator()

    def turn_on_face_slider():
        face_match_slider_control.slider.disabled = False
        face_match_slider_control.slider_amount.disabled = False

    def turn_off_face_slider():
        face_match_slider_control.slider.disabled = True
        face_match_slider_control.slider_amount.disabled = True

    check_faces_button_control = Button("Check Faces", False, page, on_click_extra_on=turn_on_face_slider,
                                        on_click_extra_off=turn_off_face_slider)
    check_faces_button = check_faces_button_control.button_creator()

    track_bodies_control = Button("Track Bodies", False, page)
    track_bodies_button = track_bodies_control.button_creator()

    seed_enter_number_control = EnterNumber("Seed", -1, data_type=int, page=page)
    seed_enter_number = seed_enter_number_control.enter_number_creator()

    prompt_textfield_button_control = TextFieldButton(button_text="+Add Prompt", field_name="Prompt", on_submit=lambda x: person_button_adder(x), on_click=lambda x: person_button_adder(x))
    prompt_textfield_button = prompt_textfield_button_control.textfield_button_creator()

    def check_skip_if_exists(event=None):
        save = save_folder_textfield_button_control
        save.textfield.error_text = None
        page.update(save.textfield)

        if skip_if_exists_button_control.button.text == "ON" and save.textfield.value == "":
            save.textfield.error_text = "Skip If Exists Needs A Directory To Be Set"
            page.update(save.textfield)
            return ""

        elif skip_if_exists_button_control.button.text == "OFF":
            return ""

        else:
            return save.textfield.value

    def get_save_folder(event):
        if event.path:
            save_folder_textfield_button_control.textfield.value = event.path
        check_skip_if_exists()
        page.update(save_folder_textfield_button_control.textfield)

    save_folder_picker = ft.FilePicker(on_result=lambda x: get_save_folder(x))

    save_folder_textfield_button_control = TextFieldButton(default_field_val=default_save_folder, button_text="Browse", field_name="Image Save Folder", hint_text="Enter Save Folder")
    save_folder_textfield_button_control.button.on_click = lambda x: save_folder_picker.get_directory_path(initial_directory=output_path)
    save_folder_textfield_button_control.textfield.on_change = check_skip_if_exists

    save_folder_textfield_button = save_folder_textfield_button_control.textfield_button_creator()
    save_folder_textfield_button.padding = ft.padding.only(17)

    def get_dir():
        global exit_loop
        exit_loop = True
        file_picker.get_directory_path()

    file_picker = ft.FilePicker(on_result=lambda x: get_file_source(x))

    add_video_control = SourceButton("+Add Video")
    add_video = add_video_control.source_button_creator()
    add_video_control.button.on_click = lambda x: file_picker.pick_files(file_type=ft.FilePickerFileType.VIDEO)

    add_image_folder_control = SourceButton("+Add Image Folder")
    add_image_folder = add_image_folder_control.source_button_creator()
    add_image_folder_control.button.on_click = lambda x: get_dir()
    source_link_control = TextFieldButton("Add Source", "Source", "Enter Source Here", on_click=lambda x: get_source)

    def update_ui_image_source(path):
        global video_fps

        image_container_control.source_control.textfield.value = path
        if os.path.isdir(path) and video_fps is None:
            image_container_control.fps_control.textfield.error_text = "Using Default Fps"
        page.update()

    def update_image_source(path):
        global exit_loop
        global image_source

        if os.path.isdir(path):
            exit_loop = True
            time.sleep(.1)
            exit_loop = False

            image_source = path
            update_ui_image_source(path)
            images = [file for file in glob.glob(path + "\*") if file.endswith(image_types)]

            if images:
                source_picker.open = False
                source_link_control.textfield.helper_text = f"Success! Found {len(images)} Images In Folder"
                page.update(source_link)
                page.update(source_picker)

                while not generating and not exit_loop:
                    deltas = deque(maxlen=5)
                    deltas.append(0) if len(deltas) == 0 else None
                    delta = 0

                    for file in images:

                        if exit_loop or generating:
                            break

                        if path != image_source:
                            exit_loop = True
                            time.sleep(.1)
                            print("hello")
                            break

                        before = time.perf_counter()
                        image_container_control.image.src_base64 = get_base64(file)
                        wait_time = frame_to_time(show_rate)
                        new_wait_time = wait_time - mean(deltas)
                        right_before = time.perf_counter() - before
                        actual_wait_time = new_wait_time - right_before

                        if actual_wait_time < 0:
                            actual_wait_time = 0.001

                        time.sleep(actual_wait_time)
                        after = time.perf_counter() - before

                        if after > wait_time:
                            delta = after - wait_time

                        deltas.append(delta)
                        page.update(image_container_control.image)
            else:
                source_link_control.textfield.helper_text = None
                source_link_control.textfield.error_text = image_error_text
                page.update(source_link_control.textfield)

    def get_source(event=None, path=""):
        nonlocal image_source
        global show_rate
        global exit_loop
        global video_fps
        global video_source
        global url_source
        exit_loop = False

        if not path:
            path = source_link_control.textfield.value

        source_link_control.textfield.helper_text = None
        source_link_control.textfield.error_text = None

        if os.path.isdir(path):

            if path == image_source:
                source_link_control.textfield.error_text = "Source Is Already Added"
                source_link_control.textfield.update()
                return

            update_image_source(path)

        elif os.path.isfile(path) and path.endswith(video_types):

            if path == video_source:
                source_link_control.textfield.error_text = "Source Is Already Added"
                source_link_control.textfield.update()
                return

            valid_chars = f"-_(), {string.ascii_letters}{string.digits}"
            safe_dir_name = "".join(letter for letter in os.path.basename(path) if letter in valid_chars)[:30]
            save_dir = os.path.join(images_from_video_folder, safe_dir_name)

            video_cv2 = cv2.VideoCapture(path)
            video_fps = video_cv2.get(cv2.CAP_PROP_FPS)
            frame_count = int(video_cv2.get(cv2.CAP_PROP_FRAME_COUNT))
            zfill = str(len(str(frame_count))).zfill(2)

            save_path = os.path.join(save_dir, f'%{zfill}d.png')

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            else:
                files_in_folder = len(os.listdir(save_dir))
                if files_in_folder in [frame_count-1, frame_count, frame_count+1]:
                    source_link_control.textfield.error_text = "Source Is Already Added"
                    source_link_control.textfield.update()

                    if video_source != path:
                        video_source = path
                        update_image_source(path=save_dir)
                    return

            video = ffmpeg.input(path)
            video_or_download_progress.height = 15
            video_or_download_progress.update()
            ffmpeg.output(video, save_path).run()
            video_source = path
            video_or_download_progress.height = 0
            image_container_control.fps_control.textfield.value = round(video_fps, 2)
            image_container_control.source_control.textfield.value = path
            update_image_source(save_dir)
            page.update()

        else:
            download_dir = os.path.join(output_path, "Downloaded Videos")
            download_path = download_dir+r"\%(title)s%(id)s.%(ext)s"

            with YoutubeDL({'outtmpl': f"{download_path}"}) as ydl:
                info = ydl.extract_info(f'{path}', download=False)
                filename = ydl.prepare_filename(info)
                already_downloaded = os.path.exists(filename)

                video_or_download_progress.height = 15
                video_or_download_progress.update()

                if already_downloaded:

                    source_link_control.textfield.error_text = "Already Downloaded This Link"
                    source_link_control.textfield.update()
                    video_or_download_progress.height = 0
                    video_or_download_progress.update()
                    get_source(path=filename)

                else:
                    try:
                        ydl.extract_info(f'{path}', download=True)
                        url_source = path
                        source_link_control.textfield.value = filename
                        source_link_control.textfield.update()
                        get_source(path=filename)

                    except yt_dlp.DownloadError:
                        source_link_control.textfield.error_text = "Could Not Download Link"
                        source_link_control.textfield.update()

            video_or_download_progress.height = 0
            video_or_download_progress.update()

    source_link_control.button.on_click = get_source
    source_link = source_link_control.textfield_button_creator()

    def link_update(event=None):
        clipboard = page.get_clipboard()
        if source_link_control.textfield.value == clipboard:
            get_source()

        else:
            source_link_control.textfield.value = clipboard
            source_link_control.textfield.focus()
            source_link_control.textfield.update()

    add_link_control = SourceButton("+Add Link")
    add_link = add_link_control.source_button_creator()
    add_link_control.button.on_click = link_update

    source_row = ft.Row([add_video, add_image_folder, add_link])
    button_container = ft.Container(source_row)
    video_or_download_progress = ft.ProgressBar(height=0)
    source_link_layout = ft.Column([source_link, button_container, video_or_download_progress])
    source_link_layout_container = ft.Container(source_link_layout, height=125)

    def get_file_source(event:ft.FilePickerResultEvent):

        if event.path is None and event.files is None:
            try:
                get_source(image_source)
            except NameError:
                pass

            return

        try:
            path = event.files[0].path
        except TypeError:
            path = event.path

        source_link_control.textfield.value = path
        source_link_control.textfield.update()

        get_source()

    videos_save_folder = os.path.join(output_path, 'videos')

    def open_folder_windows(event):
        os.startfile(videos_save_folder)

    completion_dialog_text = ft.Text(value=f"video saved to the {videos_save_folder} folder")
    completion_dialog_button = ft.ElevatedButton(text="OPEN FOLDER", style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=5), bgcolor=ft.colors.GREEN, color="white"), height=65, on_click=open_folder_windows)
    completion_dialog_button_container = ft.Container(completion_dialog_button)
    completion_dialog_content = ft.Column([completion_dialog_text, completion_dialog_button_container], horizontal_alignment=ft.CrossAxisAlignment.STRETCH, height=65)
    source_picker = ft.AlertDialog(title=ft.Text("ADD SOURCE", font_family="Main"), content=source_link_layout_container, shape=ft.RoundedRectangleBorder(radius=5), on_dismiss=get_source)
    completion_dialog = ft.AlertDialog(title=ft.Text("VIDEO COMPLETE", font_family="Main"),
                                       content=completion_dialog_content,
                                       shape=ft.RoundedRectangleBorder(radius=5), on_dismiss=get_source)

    def source_picker_open(event=None):
        source_picker.open = True
        page.update()

    image_container_control = ImageContainer(page, on_click=source_picker_open)
    image_container = image_container_control.image_container_creator()

    retry_amount_slider_control = Slider(name="Retry Amt", min_val=1, max_val=20, page=page, divisions=19, data_type=int, starting_val=1, small_num=True)
    retry_amount_slider = retry_amount_slider_control.slider_creator()

    similar_trio_control = Trio("Similar Retry", ["OFF", "Variation", "Seed"], page, "Change Amt", "Var Change", 0, 100, 0, 1, 100, 100, float, float, 81, 0, "OFF", ["Seed", "Variation"], ["Variation"], retry_amount_slider_control)
    similar_trio = similar_trio_control.trio_creator()

    frame_gen_trio_control = Trio("Frame Gen", ["OFF", "Auto", "All"], page, "Change Amt", "Max Frame Skip", 0, 100, 0, 6,
                        100, 6, float, int, 81, 2, "OFF", ["Auto"], ["Auto"])
    frame_gen_trio = frame_gen_trio_control.trio_creator()

    inject_trio_control = Trio("Blend Last Image", ["OFF", "ON"], page, "Blur Amt", "Opacity", 0, 50, 0, 1,
                          100, 100, int, float, 3, 0.1, "OFF", ["ON"], ["ON"])
    inject_trio = inject_trio_control.trio_creator()

    trio_column = ft.Column([similar_trio, frame_gen_trio, inject_trio], spacing=0)
    trio_container = ft.Container(trio_column, padding=ft.padding.only(7), alignment=ft.alignment.center_right)

    person_prompts_row = ft.Row(scroll=ft.ScrollMode.ALWAYS, wrap=True, spacing=0, width=2000)
    person_prompts_container = ft.Container(person_prompts_row, bgcolor=ft.colors.BLACK12, height=100, border_radius=5, margin=ft.margin.only(bottom=10), padding=0)
    error_text = ft.Text("There Was An Error")

    def close_banner(event):
        page.banner.open = False
        page.update()

    page.banner = ft.Banner(
        bgcolor=ft.colors.AMBER_100,
        leading=ft.Icon(ft.icons.WARNING_AMBER_ROUNDED, color=ft.colors.AMBER, size=40),
        content=error_text,
        actions=[
            ft.TextButton("Retry", on_click=close_banner),
            ft.TextButton("Ignore", on_click=close_banner),
            ft.TextButton("Cancel", on_click=close_banner),
        ],
    )

    def show_banner():
        page.banner.open = True
        page.update()
        time.sleep(2)
        page.banner.open = False
        page.update()

    class ValueGetter:
        def value_tester(self, control_to_test, control_name):
            value = control_to_test.value
            if value:
                return value
            else:
                error_text.value = f"To generate you need a prompt, and source"
                show_banner()
                source_picker_open()
                return False

    def remover(person_button):
        person_prompts_row.controls.remove(person_button)
        page.update()

    def person_button_adder(event):
        prompt_text = prompt_textfield_button_control.textfield.value
        prompt_textfield_button_control.textfield.value = ""
        prompt_textfield_button_control.textfield.focus()
        person_button_control = PromptButton(prompt_text)
        person_button = person_button_control.prompt_button_creator()
        person_button_control.button.on_click = lambda x: remover(person_button)
        person_prompts_row.controls.append(person_button)
        page.update()
        return person_button

    def get_prompts():
        prompt_list = []

        for control in person_prompts_row.controls:
            prompt = control.content.text
            prompt_list.append(prompt)

        if not prompt_list:
            error_text.value = "No Added Prompts"
            show_banner()
            page.update()

        return prompt_list if prompt_list is not None else False

    def completed():
        image_container_control.progress_bar.bar_height = 0
        completion_dialog.open = True
        page.update()

    def show_generation_preview(folder: str):
        global generating
        killed = False

        while generating:

            if image_container_control.progress_bar.bar_height == 0:
                image_container_control.progress_bar.bar_height = 10
                image_container_control.progress_bar.update()

            time.sleep(.5)

            def get_files():
                files = [file for file in glob.glob(folder+"\*") if file.endswith(image_types)]
                return files

            files = get_files()
            if len(files) == 0:
                time.sleep(1)
                continue

            all_files = len(files)
            max_speed = 0.0000001
            min_speed = 0.001
            delta = min_speed - max_speed
            divisions = delta/all_files

            images = {}

            for i, file in enumerate(files):

                if not generating:
                    image_container_control.progress_bar.bar_height = 0
                    completed()
                    killed = True
                    break

                i += 1
                wait_time = divisions*i

                image = images.get(file)
                if image is None:
                    image = get_base64(file)
                    images.update({file: image})

                image_container_control.image.src_base64 = image
                time.sleep(wait_time)
                image_container_control.image.update()

        if not generating and not killed:
            completed()

    def generate(event):
        global generating
        global image_source

        value_getter = ValueGetter()
        image_folder = image_source
        check_faces = check_faces_button_control.button.text == "ON"
        skip_if_exists = check_skip_if_exists()
        steps = int(steps_slider_control.slider_amount.value)
        cfg = int(cfg_slider_control.slider_amount.value)
        height = int(height_slider_control.slider_amount.value)
        denoising = float(denoising_slider_control.slider_amount.value)
        seed = int(seed_enter_number_control.textfield.value)
        restore_faces = restore_faces_button_control.button.text == "ON"
        width = int(width_slider_control.slider_amount.value)
        face_match_sensitivity = face_match_slider_control.slider.value
        func_prompt_list = get_prompts()
        inject_last_image = {"Blur": inject_trio_control.slider_one_control.slider.value, "Opacity": inject_trio_control.slider_two_control.slider.value} if inject_trio_control.dropdown_control.dropdown.value == "ON" else False
        loopback = int(loopback_slider_control.slider_amount.value)
        variation_scaling = similar_trio_control.slider_two_control.slider.value if similar_trio_control.dropdown_control.dropdown.value == "Variation" else 0
        ssim_retry = int(similar_trio_control.extra_control.slider.value) if similar_trio_control.dropdown_control.dropdown.value != "OFF" else 1
        ssim_threshold = similar_trio_control.slider_one_control.slider.value if similar_trio_control.dropdown_control.dropdown.value != "OFF" else False
        seed_scaling = True if similar_trio_control.dropdown_control.dropdown.value == "Seed" else False
        auto_skip_frames = frame_gen_trio_control.dropdown_control.dropdown.value.lower() == "Auto".lower()
        frame_gen_all = frame_gen_trio_control.dropdown_control.dropdown.value.lower() == "All".lower()
        frame_gen_auto = frame_gen_trio_control.dropdown_control.dropdown.value.lower() == "Auto".lower()
        frame_gen_skip_ratio = frame_gen_trio_control.slider_one_control.slider.value if frame_gen_trio_control.dropdown_control.dropdown.value == "Auto" else 100
        fps = image_container_control.fps_control.textfield.value
        sampler = sampler_dropdown_control.dropdown.value
        track_bodies = track_bodies_control.button.text == "ON"
        pprint.pprint(locals())

        if all([func_prompt_list, image_folder]):
            generating = True

            def this_func():
                global generating
                time.sleep(1)
                print("starting")
                interrupt_button_container.expand = 1
                interrupt_button_container.width = None
                page.update()

                status = track.test_main(image_folder=image_folder, skip_if_exists=skip_if_exists, steps=steps,
                                face_match_sensitivity=face_match_sensitivity, func_prompt_list=func_prompt_list,
                                inject_last_image=inject_last_image, loopback=loopback,
                                variation_scaling=variation_scaling,
                                ssim_retry=ssim_retry, seed_scaling=seed_scaling,
                                stable_diffusion_folder=stable_diffusion_folder,
                                check_faces=check_faces, default_save_folder=default_save_folder,
                                output_path=output_path,
                                frame_gen_all=frame_gen_all, frame_gen_auto=frame_gen_auto, sampler=sampler, fps=fps,
                                prompt_by_body=track_bodies, ssim_threshold=ssim_threshold,
                                frame_gen_skip_ratio=frame_gen_skip_ratio, cfg=cfg, height=height, width=width,
                                denoising=denoising, restore_faces=restore_faces, seed=seed, pipe=stable_diffusion_conn,server=server)

                generating = False
                interrupt_button_container.expand = 0
                interrupt_button_container.width = 0
                page.update()

            if skip_if_exists:
                save_folder = os.path.join(output_path, skip_if_exists)
                threading.Thread(target=this_func).start()
                show_generation_preview(save_folder)
            else:
                show_generation_preview(default_save_folder)

    left_column = ft.Column([image_container, person_prompts_container, prompt_textfield_button], expand=True)
    left_container = ft.Container(content=left_column, expand=6)

    layout_row = ft.Row([sampler_dropdown, seed_enter_number, loopback_slider, face_match_slider])
    row_container = ft.Container(layout_row, padding=ft.padding.only(left=20, top=20))

    button_row = ft.Row([restore_faces_button, skip_if_exists_button, track_bodies_button, check_faces_button], expand=True)
    button_container = ft.Container(content=button_row, padding=ft.padding.only(left=20), alignment=ft.alignment.top_center)
    generate_text = ft.Text("GENERATE", font_family="Main", size=35)
    generate_button = ft.ElevatedButton(content=generate_text, style=ft.ButtonStyle(bgcolor=ft.colors.GREEN, color=ft.colors.WHITE, shape=ft.RoundedRectangleBorder(radius=10)),
                                 height=150, on_click=generate, expand=2)
    generate_button_container = ft.Container(generate_button, expand=2)

    def send_message(message, event=None):
        ui.send(message)

    interrupt_button = ft.ElevatedButton(content=ft.Icon(ft.icons.CANCEL), icon_color="white", style=ft.ButtonStyle(bgcolor=ft.colors.RED, color=ft.colors.WHITE, shape=ft.RoundedRectangleBorder(radius=10)),
                                 height=150, on_click=lambda x:send_message("STOP"), expand=1)
    interrupt_button_container = ft.Container(interrupt_button, expand=0, width=0)
    generate_and_interrupt = ft.Row([generate_button_container, interrupt_button_container])
    generate_container = ft.Container(generate_and_interrupt, alignment=ft.alignment.bottom_center)

    right_column = ft.Column([completion_dialog, source_picker, steps_slider, width_slider, height_slider, cfg_slider, denoising_slider, row_container, button_container, trio_container, save_folder_textfield_button], spacing=0, scroll=ft.ScrollMode.AUTO, height=1000, expand=True)
    generate_column = ft.Column([right_column, generate_container])
    right_container = ft.Container(content=generate_column, expand=4, padding=ft.padding.only(top=30, right=2))

    main_ui_container = ft.Container(ft.Row([left_container, right_container]), expand=True)
    page.overlay.append(save_folder_picker)
    page.overlay.append(file_picker)
    page.add(main_ui_container)


ft.app(target=main, assets_dir="Assets")

