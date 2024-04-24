import os
from modal import App, Image, Volume, NetworkFileSystem, Mount, enter, method, build
from sympy import im

from modal import web_endpoint
from makevideo_parallel import process_line, process_short_line, process_qa_line, prepare_tasks, CommandRunner


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

# vol = Volume.from_name("arxiv-volume", create_if_missing=True)
volume = NetworkFileSystem.from_name("arxiv-nfs", create_if_missing=True)

# volume.
# volume.add_local_dir("/workspaces/ArxivPapers/.temp/1910.13461_files/", ".temp/1910.13461_files/")


@app.cls(
    cpu=8,
    image=app_image,
    network_file_systems={"/root/arxivvideo": volume},
    mounts=[
        Mount.from_local_dir(
            "/workspaces/ArxivPapers/.temp/1910.13461_files/", remote_path="/root/.temp/1910.13461_files"
        )
    ],
    timeout=60 * 60,
)
class ArxivVideo:

    # @build()  # add another step to the image build
    def builder(self):
        import nltk

        nltk.download("punkt")

        import spacy
        from spacy.cli import download

        download("en_core_web_lg")

    @method()
    def process_line_f(self, i, line, dr, args, block_coords, gptpagemap):
        i, mp4_file, ex = process_line(i, line, dr, args, block_coords, gptpagemap)

        data = b""
        with open(mp4_file, "rb") as f:
            for chunk in f:
                data += chunk
            # volume.write_file(os.path.join(dr, mp4_file), f)

        return i, mp4_file, data, ex

    @method()
    def process_short_line_f(self, i, line, page_num, dr, args):
        i, mp4_file, ex = process_short_line(i, line, page_num, dr, args)

        data = b""
        with open(mp4_file, "rb") as f:
            for chunk in f:
                data += chunk
            # volume.write_file(os.path.join(dr, mp4_file), f)

        return i, mp4_file, data, ex

    @method()
    def process_qa_line_f(self, i, line, page_num, dr, args) -> tuple[int, str, bytes, Exception]:
        i, mp4_file, _, ex = process_qa_line(i, line, page_num, dr, args)

        data = b""
        with open(mp4_file, "rb") as f:
            for chunk in f:
                data += chunk
            # volume.write_file(os.path.join(dr, mp4_file), f)

        return i, mp4_file, data, ex

    @method()
    def makevideo(self, args, video_type: str):
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

        print(args, video_type)

        files = glob.glob(os.path.join("/root/.temp", f"{args.paperid}_files", "zipfile.zip"))
        print(files)

        if files:
            zip_name = max(files, key=os.path.getmtime)
            dr = os.path.join(".temp", f"{args.paperid}_files", "output")
        else:
            return

        if os.path.exists(dr):
            shutil.rmtree(dr)

        with zipfile.ZipFile(zip_name, "r") as zipf:
            zipf.extractall(dr)

        mp4_main_list = os.path.join(dr, "mp4_list.txt")
        mp4_main_output = os.path.join(dr, "output.mp4")

        short_mp3s = os.path.join(dr, "shorts_mp3_list.txt")
        mp4_short_list = os.path.join(dr, "short_mp4_list.txt")
        mp4_short_output = os.path.join(dr, "output_short.mp4")

        qa_mp3_list = os.path.join(dr, "qa_mp3_list.txt")
        mp4_qa_list = os.path.join(dr, "qa_mp4_list.txt")
        mp4_qa_output = os.path.join(dr, "output_qa.mp4")

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
        for page_num, _ in pages.items():
            page_num_filename_no_ext = os.path.join(dr, str(page_num))
            page_num_pdf = f"{page_num_filename_no_ext}.pdf"
            logfile_path = os.path.join(dr, "logs", f"pdf.log")

            with CommandRunner(logfile_path, args) as run:
                run.extract_page_as_pdf(os.path.join(dr, "main.pdf"), page_num + 1, page_num_pdf)

        # =============== MAIN VIDEO ====================

        if video_type == "long":
            block_coords = pickle.load(open(os.path.join(dr, "block_coords.pkl"), "rb"))
            gptpagemap = pickle.load(open(os.path.join(dr, "gptpagemap.pkl"), "rb"))

            results = list(
                self.process_line_f.starmap(
                    [(i, line, dr, args, block_coords, gptpagemap) for i, line in enumerate(lines)]
                )
            )

            results.sort(key=lambda x: x[0])

            mp4_list = mp4_main_list

        elif video_type == "short":
            with open(short_mp3s, "r") as f:
                lines = f.readlines()

            results = list(self.process_short_line_f.starmap([(i, line, i, dr, args) for i, line in enumerate(lines)]))
            results.sort(key=lambda x: x[0])

            mp4_list = mp4_short_list

        elif video_type == "qa":
            with open(qa_mp3_list, "r") as f:
                lines = f.readlines()

            # lines = [lines[0], lines[1]]
            print(lines)

            qa_pages = pickle.load(open(os.path.join(dr, "qa_pages.pkl"), "rb"))
            tasks = prepare_tasks(dr, lines, qa_pages, args)
            print(tasks)

            results = list(self.process_qa_line_f.starmap(tasks))
            results.sort(key=lambda x: x[0])

            mp4_list = mp4_qa_list

        with open(mp4_list, "w") as mp4_list_f:
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
                    mp4_list_f.write(f"file {os.path.basename(mp4_file)}\n")
        import subprocess

        command, mp4_output = self.create_video_command(args, mp4_list, f"output_{video_type}.mp4")
        print(command)
        # proc_result = subprocess.run(command, shell=True)
        proc_result = os.system(command)
        if True:  # proc_result.returncode == 0:
            with open(mp4_output, "rb") as f:
                data = f.read()
                return data, video_type
        else:
            raise Exception("Error creating video")

    def create_video_command(self, args, mp4_list, mp4_output):
        return f"{args.ffmpeg} -f concat -i {mp4_list} -y -c copy {mp4_output}", mp4_output


# @stub.function(image=app_image)
# @web_endpoint()
# def get_storage(key: str):
#     return timecop_storage.get(key) if timecop_storage.contains(key) else {}


@app.local_entrypoint()
def main():
    import argparse

    paperid = "1910.13461"
    args = argparse.Namespace(paperid=paperid, gs="gs", ffmpeg="ffmpeg", ffprobe="ffprobe")
    arx = ArxivVideo()

    video_args = [(args, "long"), (args, "short"), (args, "qa")]
    results = list(arx.makevideo.starmap(video_args))

    for data, video_type in results:
        with open(f".temp/{paperid}/{paperid}_{video_type}.mp4", "wb") as f:
            f.write(data)
