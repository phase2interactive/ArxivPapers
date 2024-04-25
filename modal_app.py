import os
import argparse
from modal import App, Image, Volume, NetworkFileSystem, Mount, Secret, enter, method, build, web_endpoint
from sympy import im

from main import main as main_func
from makevideo_parallel import process_line, process_short_line, process_qa_line, prepare_tasks, CommandRunner

import pickle
import argparse
import openai
from tex.utils import *
from htmls.utils import *
from map.utils import *
from gpt.utils import *
from speech.utils import *
from zip.utils import *
from gdrive.utils import *
import random
from pathlib import Path


def builder():
    import nltk

    nltk.download("punkt")

    import spacy
    from spacy.cli import download

    download("en_core_web_lg")


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")


app = App("arxiv")

# app_image = modal.Image.from_dockerfile("Dockerfile.dev", context_mount=modal.Mount.from_local_file("requirements.txt"))
app_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "ffmpeg",
        "texlive-latex-base",
        "texlive-latex-extra",
        "texlive-fonts-extra",
        "texlive-science",
        "latex2html",
        "texlive-publishers",
        "apt-transport-https",
        "ca-certificates",
        "gnupg",
        "curl",
    )
    .pip_install_from_requirements("requirements.txt")
    .run_function(builder)
    .copy_local_dir("/workspaces/ArxivPapers/imgs", "/root/imgs")
    # .run_commands(
    #     "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg",
    #     "echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' > /etc/apt/sources.list.d/google-cloud-sdk.list",
    #     "apt-get update && apt-get install -y google-cloud-cli",
    # )
)

volume = Volume.from_name("arxiv-volume", create_if_missing=True)
VOLUME_PATH = "/root/shared"


def latex(args: argparse.Namespace):
    remove_oldfiles_samepaper(args.paperid)
    files_dir = download_paper(args.paperid)

    tex_files = get_tex_files(files_dir)
    logging.info(f"Found .tex files: {tex_files}")

    main_file = get_main_source_file(tex_files, files_dir)

    if main_file:
        logging.info(f"Found [{main_file}.tex] as a main source file")
    else:
        raise Exception("No main source file found")

    return main_file, files_dir, tex_files


@app.cls(
    cpu=8,
    image=app_image,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60,
    secrets=[Secret.from_name("llms")],
)
class ArxivVideo:

    def __init__(self, args: argparse.Namespace, video_args):
        self.args = args
        self.video_args = video_args

    # @build()  # add another step to the image build
    def builder(self):
        import nltk

        nltk.download("punkt")

        import spacy
        from spacy.cli import download

        download("en_core_web_lg")

    def inititalize_directory(self, args):

        from math import e
        from platform import system
        import zipfile
        import re
        import fitz
        from PIL import Image, ImageDraw
        import shutil
        import glob
        import pickle
        import argparse
        import subprocess
        from multiprocessing import Pool
        import cProfile
        import pstats
        from datetime import datetime
        import logging
        from typing import Any
        from collections import defaultdict
        from ffprobe import FFProbe
        import ffmpeg
        import argparse

        # local working directory
        dr = os.path.join(".temp", args.paperid, "output")
        os.makedirs(dr, exist_ok=True)

        # the zip file is on the shared volume
        zip_path = os.path.join(VOLUME_PATH, args.paperid, f"{args.paperid}.zip")

        if os.path.exists(dr):
            shutil.rmtree(dr)

        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(dr)

        file_paths = {
            "mp4_main_list": os.path.join(dr, "mp4_list.txt"),
            "mp4_main_output": os.path.join(dr, "output.mp4"),
            "short_mp3s": os.path.join(dr, "shorts_mp3_list.txt"),
            "mp4_short_list": os.path.join(dr, "short_mp4_list.txt"),
            "mp4_short_output": os.path.join(dr, "output_short.mp4"),
            "qa_mp3_list": os.path.join(dr, "qa_mp3_list.txt"),
            "mp4_qa_list": os.path.join(dr, "qa_mp4_list.txt"),
            "mp4_qa_output": os.path.join(dr, "output_qa.mp4"),
        }

        with open(os.path.join(dr, "mp3_list.txt"), "r") as f:
            lines = f.readlines()

        # filter lines to only include those that have "page4" in them
        # lines = [line for line in lines if "page0" in line]
        # print(lines)

        # Group lines by page number
        pages = defaultdict(list)
        for line in lines:
            line = line.strip()
            components = line.split()
            match = re.search(r"page(\d+)", components[1])
            page_num = int(match.group(1))
            pages[page_num].append(line)

        # output each page of the pdf this is used by downstream functions
        print(pages)
        pdfs = []
        for page_num, _ in pages.items():
            page_num_pdf = f"{os.path.join(dr, str(page_num))}.pdf"
            logfile_path = os.path.join(dr, "logs", f"pdf.log")
            main_pdf_file = os.path.join(dr, "main.pdf")

            with CommandRunner(logfile_path, args) as run:
                run.extract_page_as_pdf(main_pdf_file, page_num + 1, page_num_pdf)
                pdfs.append(page_num_pdf)

        return dr, lines, pdfs, file_paths

    @enter()
    def on_enter(self):
        self.dr, self.lines, self.pdfs, self.file_paths = self.inititalize_directory(self.args)

    @method()
    def download_and_zip(self, paperid: str):
        import shutil

        zip_file = main_func(self.args)

        volume_path = f"{VOLUME_PATH}/{paperid}/{paperid}.zip"

        # copy zip_file to volume_path
        shutil.copy(zip_file, volume_path)

        volume.commit()

        return zip_file

    @method()
    def process_line_f(self, i, line):

        block_coords = pickle.load(open(os.path.join(self.dr, "block_coords.pkl"), "rb"))
        gptpagemap = pickle.load(open(os.path.join(self.dr, "gptpagemap.pkl"), "rb"))

        i, mp4_file, ex = process_line(i, line, self.dr, self.video_args, block_coords, gptpagemap)

        data = b""
        with open(mp4_file, "rb") as f:
            for chunk in f:
                data += chunk
            # volume.write_file(os.path.join(dr, mp4_file), f)

        return i, mp4_file, data, ex

    @method()
    def process_short_line_f(self, i, line, page_num):
        i, mp4_file, _, ex = process_short_line(i, line, page_num, self.dr, self.video_args)

        if ex:
            raise ex
        else:
            data = b""
            with open(mp4_file, "rb") as f:
                for chunk in f:
                    data += chunk
                # volume.write_file(os.path.join(dr, mp4_file), f)

            return i, mp4_file, data, ex

    @method()
    def process_qa_line_f(self, line, line_num, input_path, dr, args) -> tuple[int, str, bytes, Exception]:
        i, mp4_file, _, ex = process_qa_line(line, line_num, input_path, self.dr, self.video_args)

        data = b""
        with open(mp4_file, "rb") as f:
            for chunk in f:
                data += chunk
            # volume.write_file(os.path.join(dr, mp4_file), f)

        return i, mp4_file, data, ex

    @method()
    def makevideo(self, video_type: str):
        if video_type == "long":
            results = list(
                self.process_line_f.starmap(
                    [(i, line) for i, line in enumerate(self.lines)],
                    return_exceptions=True,
                )
            )

            if any([isinstance(result, Exception) for result in results]):
                raise Exception("Error occurred")

            results.sort(key=lambda x: x[0])

        elif video_type == "short":
            with open(self.file_paths["short_mp3s"], "r") as f:
                lines = f.readlines()

            results = list(self.process_short_line_f.starmap([(i, line, i) for i, line in enumerate(lines)]))
            results.sort(key=lambda x: x[0])

        elif video_type == "qa":
            with open(self.file_paths["qa_mp3_list"], "r") as f:
                lines = f.readlines()

            # lines = [lines[0], lines[1]]

            qa_pages = pickle.load(open(os.path.join(self.dr, "qa_pages.pkl"), "rb"))
            tasks = prepare_tasks(self.dr, lines, qa_pages, self.video_args)

            results = list(self.process_qa_line_f.starmap(tasks))
            results.sort(key=lambda x: x[0])

        mp4_list = Path(self.dr) / "mp4_list.txt"
        with open(mp4_list, "w") as mp4f:
            for _, mp4_file, data, ex in results:
                if ex:
                    logging.error(f"Error occurred: {ex}")
                    return
                else:
                    # data = b""
                    # for chunk in volume.read_file(mp4_file):
                    #     data += chunk
                    with open(mp4_file, "wb") as f:
                        f.write(data)
                    print(mp4_file)
                    mp4f.write(f"file {os.path.basename(mp4_file)}\n")
        import subprocess

        command, mp4_output = self.create_video_command(self.args, mp4_list, f"output_{video_type}.mp4")
        print(command)
        proc_result = subprocess.run(command, shell=True)
        # proc_result = os.system(command)
        if proc_result.returncode == 0:
            with open(mp4_output, "rb") as f:
                data = f.read()
                return data, video_type
        else:
            raise Exception("Error creating video")

    def create_video_command(self, args, mp4_list, mp4_output):
        return f"{args.ffmpeg} -f concat -i {mp4_list} -y -c copy {mp4_output}", mp4_output

    @method()
    def run(self, paperid, video_types=["long"]) -> tuple[dict[str, bytes], bytes]:

        # zip_file = self.download_and_zip.remote(paperid)
        zip_file = f"{VOLUME_PATH}/{paperid}/{paperid}.zip"
        zip_data = open(zip_file, "rb").read()
        results = list(self.makevideo.map(video_types))

        videos = {}
        for data, video_type in results:
            videos[video_type] = data

        # volume.delete_file(zip_file)
        return (videos, zip_data)


# @stub.function(image=app_image)
# @web_endpoint()
# def get_storage(key: str):
#     return timecop_storage.get(key) if timecop_storage.contains(key) else {}


@app.local_entrypoint()
def main():

    paperid = "2310.08560"

    args = argparse.Namespace(
        paperid=paperid,
        l2h=True,
        verbose="info",
        pdflatex="pdflatex",
        latex2html="latex2html",
        latexmlc="latexmlc",
        stop_word="",
        ffmpeg="ffmpeg",
        gs="gs",
        cache_dir="cache",
        gdrive_id="",
        voice="onyx",
        final_audio_file="final_audio",
        chunk_mp3_file_list="mp3_list.txt",
        manual_gpt=False,
        include_summary=True,
        extract_text_only=False,
        create_video=True,
        create_short=True,
        create_qa=True,
        create_audio_simple=False,
        llm_strong="gpt-4-0125-preview",
        llm_base="gpt-4-0125-preview",
        openai_key=os.environ.get("OPENAI_API_KEY", ""),
        tts_client="openai",
    )

    video_args = argparse.Namespace(paperid=paperid, gs="gs", ffmpeg="ffmpeg", ffprobe="ffprobe")

    arx = ArxivVideo(args, video_args)

    images, zip = arx.run.remote(paperid, ["long", "short", "qa"])

    for k, v in images.items():
        mp4_file = f".temp/{paperid}/{paperid}_{k}.mp4"
        with open(mp4_file, "wb") as f:
            f.write(v)
            print(f"Saved {mp4_file}")
