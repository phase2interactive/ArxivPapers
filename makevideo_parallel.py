import os
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


logging.basicConfig(level=logging.INFO, format="\n%(asctime)s - %(levelname)s - %(message)s")


class CommandRunner:
    def __init__(self, logfile_name, args):
        self.logfile_name = logfile_name
        self.args = args
        os.makedirs(os.path.dirname(self.logfile_name), exist_ok=True)

    def __enter__(self):
        self.logfile = open(self.logfile_name, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logfile.close()

    def pdf_to_png(self, pdf_file, png_file, r="-r500") -> subprocess.CompletedProcess[bytes]:
        return self.run(
            f"{self.args.gs} -sDEVICE=png16m {r} -o {png_file} {pdf_file}",
            shell=True,
        )

    def extract_page_as_pdf(self, pdf_file, page_num, output_file) -> subprocess.CompletedProcess[bytes]:
        command = [
            self.args.gs,
            "-sDEVICE=pdfwrite",
            "-dNOPAUSE",
            "-dBATCH",
            f"-dFirstPage={page_num}",
            f"-dLastPage={page_num}",
            f"-sOutputFile={output_file}",
            pdf_file,
        ]
        return self.run(command)

    def create_video_from_image_and_audio(
        self, png_file, mp3_file, resolution, video_file
    ) -> subprocess.CompletedProcess[bytes]:
        command = f"{self.args.ffmpeg} -loop 1 -i {png_file} -i {mp3_file} -vf {resolution} -c:v libx264 -tune stillimage -y -c:a aac -b:a 128k -pix_fmt yuv420p -shortest {video_file}"
        return self.run(command, check=False, shell=True)

    def run(self, command, check=True, **kwargs) -> subprocess.CompletedProcess[bytes]:

        if kwargs.get("shell", False):
            cmd_text = f"{command}\n"
        else:
            cmd_text = f"{' '.join(command)}\n"
        self.logfile.write(cmd_text)
        if kwargs.get("capture_output", False):
            ret_val = subprocess.run(command, check=check, **kwargs)
        else:
            ret_val = subprocess.run(command, stdout=self.logfile, stderr=self.logfile, check=check, **kwargs)
        if ret_val.returncode != 0:
            logging.error(f"\t >> Non-zero return code running command: {cmd_text}")
            logging.error(f"\t >> Return code: {ret_val.returncode}")
        else:
            logging.info(f"\t >> Command {cmd_text} ran successfully")
        return ret_val

    def __call__(self, command, check=True, **kwargs):
        return self.run(command, check, **kwargs)


def process_line(i, line, dr, args) -> tuple[Any | int, str]:
    print([i, line])
    # Remove the newline character at the end of the line
    line = line.strip()

    # Split the line into components
    components = line.split()

    # The filename is the second component
    audio = components[1].replace(".mp3", "")
    video = audio.replace("-", "")

    video_file = f"{os.path.join(dr, video)}.mp4"
    final_video_file = f"{os.path.join(dr, video)}_final.mp4"

    # if final_video_file exists
    if os.path.exists(final_video_file):
        return i, final_video_file

    # The number is the fourth component (without the #)
    match = re.search(r"page(\d+)", components[1])
    page_num = int(match.group(1))
    page_num_filename_no_ext = os.path.join(dr, str(page_num))
    page_num_png = f"{page_num_filename_no_ext}_{i}.png"
    page_num_pdf = f"{page_num_filename_no_ext}.pdf"

    logfile_path = os.path.join(dr, "logs", f"{i}{video}.log")
    with CommandRunner(logfile_path, args) as run:
        # convert to PNG
        run.pdf_to_png(page_num_pdf, page_num_png, "-r300")

        if "summary" not in components[1]:
            doc = fitz.open(page_num_pdf)
            page = doc[0]

            match = re.search(r"block(\d+)", components[1])
            block_num = int(match.group(1))

            for pb in gptpagemap:
                if isinstance(pb, list):
                    if pb[1] == page_num and pb[2] == block_num:
                        coords = block_coords[pb[0]][block_num]
                        break

            # Load an image
            image = Image.open(page_num_png)

            # Calculate scale factors
            scale_x = image.width / page.rect.width
            scale_y = image.height / page.rect.height

            # Rescale coordinates
            x0, y0, x1, y1 = coords
            x0 *= scale_x
            y0 *= scale_y
            x1 *= scale_x
            y1 *= scale_y

            # find out if this rectangle is on the left, right or whole width
            if coords[2] > page.rect.width / 2:
                pointerleft = Image.open("imgs/pointertoleft.png")
                pointer = pointerleft.convert("RGBA")
                onrightside = True
            else:
                pointerright = Image.open("imgs/pointertoright.png")
                pointer = pointerright.convert("RGBA")
                onrightside = False

            # Create draw object
            draw = ImageDraw.Draw(image)

            # Define thickness
            thickness = 5

            # Draw several rectangles to simulate thickness
            for i in range(thickness):
                draw.rectangle([x0 - i - 5, y0 - i - 5, x1 + i + 5, y1 + i + 5], outline="green")

            # Calculate the center of the rectangle
            rect_center_y = (y0 + y1) / 2

            # Scale down the pointer image while preserving aspect ratio
            desired_height = image.height / 20
            aspect_ratio = pointer.width / pointer.height
            new_width = int(aspect_ratio * desired_height)
            pointer = pointer.resize((new_width, int(desired_height)))

            # Calculate position for the pointer
            if onrightside:
                pointer_x0 = x1 + 20
            else:
                pointer_x0 = x0 - 20 - new_width

            pointer_y0 = rect_center_y - (pointer.height / 2)

            # Paste the pointer on the main image
            image.paste(pointer, (int(pointer_x0), int(pointer_y0)), pointer)  # The last argument is for transparency

            # Save the combined image to a file
            image.save(page_num_png)

        # process each image-audio pair to create video chunk

        run.create_video_from_image_and_audio(
            png_file=page_num_png,
            mp3_file=f"{os.path.join(dr, audio)}.mp3",
            resolution="scale=1920:-2",
            video_file=video_file,
        )

        # Get audio duration
        result = run(
            f"{args.ffprobe} -i {os.path.join(dr, audio)}.mp3 -show_entries format=duration -v quiet -of csv=p=0",
            check=False,
            capture_output=True,
            text=True,
            shell=True,
        )
        audio_duration = int(float(result.stdout.strip())) + 1 if result.stdout.strip() else 0
        # print(f"!!!! audio_duration = {audio_duration}")
        if audio_duration == 0:
            logging.warning(f"Audio duration is 0 for {audio}.mp3")

        # # ensure that there is no silence at the end of the video, and video len is the same as audio len
        run([args.ffmpeg, "-i", video_file, "-t", str(audio_duration), "-y", "-c", "copy", final_video_file])

        return i, final_video_file


def main(args):
    global block_coords, gptpagemap

    files = glob.glob(os.path.join(".temp", f"{args.paperid}_files", "zipfile.zip"))
    print(files)

    if files:
        zip_name = max(files, key=os.path.getmtime)
        dr = os.path.join(f"{args.paperid}_files", "output")
    else:
        return

    # if os.path.exists(dr):
    #     shutil.rmtree(dr)

    # with zipfile.ZipFile(zip_name, "r") as zipf:
    #     zipf.extractall(dr)

    with open(os.path.join(dr, "mp3_list.txt"), "r") as f:
        lines = f.readlines()

    # filter lines to only include those that have "page4" in them
    # lines = [line for line in lines if "page0" in line]
    print(lines)

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

    block_coords = pickle.load(open(os.path.join(dr, "block_coords.pkl"), "rb"))
    gptpagemap = pickle.load(open(os.path.join(dr, "gptpagemap.pkl"), "rb"))

    num_workers = 4  # min(len(lines), os.cpu_count() or 4)

    with Pool(num_workers) as pool:
        results = pool.starmap(process_line, [(i, line, dr, args) for i, line in enumerate(lines)])

    results.sort(key=lambda x: x[0])
    with open(os.path.join(dr, "mp4_list.txt"), "w") as outvideo:
        outvideo.writelines([f"file {os.path.basename(mp4_file)}\n" for _, mp4_file in results])

    # =============== SHORT VIDEO ====================
    short_mp3s = os.path.join(dr, "shorts_mp3_list.txt")

    if os.path.exists(short_mp3s):
        with open(short_mp3s, "r") as f:
            lines = f.readlines()

        with Pool(num_workers) as pool:
            results = pool.starmap(
                process_short_line, [(page_num, line, page_num, dr, args) for (page_num, line) in enumerate(lines)]
            )

        results.sort(key=lambda x: x[0])
        mp4_short_list = os.path.join(dr, "short_mp4_list.txt")
        mp4_short_output = os.path.join(dr, "output_short.mp4")

        print(mp4_short_output)
        with open(mp4_short_list, "w") as outvideo:
            outvideo.writelines([f"file {os.path.basename(mp4_file)}\n" for _, mp4_file in results])

    # =============== QA VIDEO ====================

    # if os.path.exists(os.path.join(dr, "qa_mp3_list.txt")):
    #     with open(os.path.join(dr, "qa_mp3_list.txt"), "r") as f:
    #         lines = f.readlines()

    #     qa_pages = pickle.load(open(os.path.join(dr, "qa_pages.pkl"), "rb"))

    #     pool = Pool()
    #     results = pool.starmap(
    #         process_qa_line, [(i, line, line_num, qa_pages, dr, args) for i, (line_num, line) in enumerate(lines)]
    #     )
    #     pool.close()
    #     pool.join()

    #     results.sort(key=lambda x: x[0])
    #     with open(os.path.join(dr, "qa_mp4_list.txt"), "w") as outvideo:
    #         outvideo.writelines([r for _, r in results])

    #     os.system(
    #         f'{args.ffmpeg} -f concat -i {os.path.join(dr, "qa_mp4_list.txt")} '
    #         f'-y -c copy {os.path.join(dr, "output_qa.mp4")}'
    #     )

    commands = [
        f'{args.ffmpeg} -f concat -i {os.path.join(dr, "mp4_list.txt")} -y -c copy {os.path.join(dr, "output.mp4")}'
    ]

    if os.path.exists(short_mp3s):
        commands.append(f"{args.ffmpeg} -f concat -i {mp4_short_list} -y -c copy {mp4_short_output}")

    with Pool(len(commands)) as p:
        p.map(os.system, commands)


def process_short_line(i, line, page_num, dr, args):
    print([i, line, page_num, dr, args])

    # Remove the newline character at the end of the line
    line = line.strip()

    # Split the line into components
    components = line.split()

    # The filename is the second component
    audio = components[1].replace(".mp3", "")
    video = audio.replace("-", "")

    print(audio, video)

    i_mp4 = f"{os.path.join(dr, video)}_{i}.mp4"
    i_final_mp4 = f"{os.path.join(dr, video)}{i}_final.mp4"
    # if i_final_mp4 exists
    # if os.path.exists(i_final_mp4):
    # return i, f"file {os.path.basename(i_final_mp4)}\n"

    # convert to PNG
    if page_num == 0:
        input_path = os.path.join(dr, str(page_num))
    else:
        input_path = os.path.join(dr, "slides", f"slide_{page_num}")

    logfile_path = os.path.join(dr, "logs", f"{i}{video}.log")
    with CommandRunner(logfile_path, args) as run:
        run.pdf_to_png(f"{input_path}.pdf", f"{os.path.join(dr, str(page_num))}.png")

        run.create_video_from_image_and_audio(
            png_file=f"{os.path.join(dr, str(page_num))}.png",
            mp3_file=f"{os.path.join(dr, audio)}.mp3",
            resolution="scale=1920:-2",
            video_file=i_mp4,
        )

        # ensure that there is no silence at the end of the video, and video len is the same as audio len
        result = run(
            f"{args.ffprobe} -i {os.path.join(dr, audio)}.mp3 -show_entries format=duration -v quiet -of csv='p=0'",
            check=False,
            capture_output=True,
            text=True,
            shell=True,
        )

        # get the audio duration
        audio_duration = int(float(result.stdout.strip())) + 1 if result.stdout.strip() else 0
        if audio_duration == 0:
            logging.warning(f"Audio duration is 0 for {audio}.mp3")

        # ensure that there is no silence at the end of the video, and video len is the same as audio len
        run([args.ffmpeg, "-i", i_mp4, "-t", str(audio_duration), "-y", "-c", "copy", i_final_mp4])

        return i, i_final_mp4


def process_qa_line(i, line, line_num, qa_pages, dr, args):
    # Remove the newline character at the end of the line
    line = line.strip()

    # Split the line into components
    components = line.split()

    # The filename is the second component
    audio = components[1].replace(".mp3", "")
    video = audio.replace("-", "")

    # convert to PNG
    if "question" in audio:  # question - get created slide
        turn = line_num // 2
        page_num = 0
        input_path = os.path.join(dr, "questions", f"question_{turn}")
    else:  # answer - get single page from paper
        turn = (line_num - 1) // 2
        p_num = qa_pages[turn][page_num]
        # extract the page from PDF
        os.system(
            f'{args.gs} -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dFirstPage={p_num+1} -dLastPage={p_num+1} -sOutputFile={os.path.join(dr, str(p_num))}.pdf {os.path.join(dr, "main.pdf")} > /dev/null 2>&1'
        )
        input_path = os.path.join(dr, f"{p_num}")

    qa_page = f"qa_page_{line_num}.png"
    subprocess.run(
        [args.gs, "-sDEVICE=png16m", "-r500", "-o", os.path.join(dr, qa_page), f"{input_path}.pdf"], check=True
    )

    resolution = "scale=1920:-2"
    subprocess.run(
        [
            args.ffmpeg,
            "-loop",
            "1",
            "-i",
            os.path.join(dr, qa_page),
            "-i",
            os.path.join(dr, audio) + ".mp3",
            "-vf",
            resolution,
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-y",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            os.path.join(dr, video) + ".mp4",
        ],
        check=True,
    )

    # ensure that there is no silence at the end of the video, and video len is the same as audio len
    audio_duration_command = [
        args.ffprobe,
        "-i",
        os.path.join(dr, audio) + ".mp3",
        "-show_entries",
        "format=duration",
        "-v",
        "quiet",
        "-of",
        'csv="p=0"',
    ]
    audio_duration = subprocess.run(audio_duration_command, stdout=subprocess.PIPE, check=True).stdout.decode().strip()
    audio_duration = str(int(float(audio_duration)) + 1)
    subprocess.run(
        [
            args.ffmpeg,
            "-i",
            os.path.join(dr, video) + ".mp4",
            "-t",
            audio_duration,
            "-y",
            "-c",
            "copy",
            os.path.join(dr, video) + "_final.mp4",
        ],
        check=True,
    )

    return i, f"file '{video}_final.mp4'\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--paperid", type=str, default="")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg")
    parser.add_argument("--ffprobe", type=str, default="ffprobe")
    parser.add_argument("--gs", type=str, default="gs")

    args = parser.parse_args()
    args = argparse.Namespace(paperid="2310.02304", gs="gs", ffmpeg="ffmpeg", ffprobe="ffprobe")

    main(args)
