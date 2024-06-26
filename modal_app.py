import os
import argparse
from modal import App, Image, Volume, NetworkFileSystem, Mount, Secret, enter, method, build, web_endpoint
from sympy import im
from collections import defaultdict

# from main import main as main_func
from makevideo_parallel import process_line, process_short_line, process_qa_line, prepare_qa_tasks, CommandRunner

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
from main3 import DocumentProcessor, try_decorator, list_files, Verbalizer, VerbalizerShort, QAVerbalizer, mock_llm_api
import random
from pathlib import Path
import itertools
from itertools import groupby
from pprint import pprint
from datetime import datetime

volume = Volume.from_name("arxiv-volume", create_if_missing=True)
VOLUME_PATH = "/root/shared"


def get_matcher():
    return Matcher("cache")


def builder():
    import nltk

    nltk.download("punkt")

    import spacy
    from spacy.cli import download

    download("en_core_web_lg")

    _ = get_matcher()


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


def latex(args: argparse.Namespace) -> tuple[str, str, list[str]]:
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


def unpack_pdf(lines: list[str], dr: str, main_pdf_file: str) -> list:
    pages = defaultdict(list)
    for line in lines:
        line = line.strip()
        components = line.split()
        match = re.search(r"page(\d+)", components[1])
        page_num = int(match.group(1))
        pages[page_num].append(line)

    # extract each page of the pdf used by downstream functions
    pdfs = []
    for page_num, _ in pages.items():
        page_num_pdf = f"{os.path.join(dr, str(page_num))}.pdf"

        ret = os.system(
            f"gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dFirstPage={page_num + 1} -dLastPage={page_num + 1} -sOutputFile={page_num_pdf} {main_pdf_file} > /dev/null 2>&1"
        )
        if ret != 0:
            raise Exception(f"Error extracting page {page_num} Return code: {ret}")
        # run.extract_page_as_pdf(main_pdf_file, page_num + 1, page_num_pdf)
        pdfs.append(page_num_pdf)
    return pdfs


def walk_zip(dir, zip_path) -> list:
    files = []
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for dirpath, _, filenames in os.walk(dir):
            for filename in filenames:
                # check for the zip path being in the dir to avoid recursion
                if filename == os.path.basename(zip_path):
                    continue
                file_path = os.path.join(dirpath, filename)
                print(file_path)
                zipf.write(file_path)
                files.append(file_path)

    return files


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

        # get a unique value based on the current time
        self.run_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        # self.run_id = "20240505194839160708"

        print(f"run_id: {self.run_id}")
        if args.gdrive_id:
            self.gdrive_client = GDrive(args.gdrive_id)

    # @build()  # add another step to the image build
    def builder(self):
        import nltk

        nltk.download("punkt")

        import spacy
        from spacy.cli import download

        download("en_core_web_lg")

        _ = Matcher("cache")

    def inititalize_directory(self, args, target_dir=".temp"):
        import zipfile
        import re
        import shutil
        from collections import defaultdict

        # local working directory
        dr = os.path.join(target_dir, args.paperid, "output")
        os.makedirs(dr, exist_ok=True)
        print("dr: ", dr)

        # the zip file is on the shared volume
        zip_path = os.path.join(VOLUME_PATH, args.paperid, f"{args.paperid}.zip")

        if os.path.exists(dr):
            shutil.rmtree(dr)

        if False:  # self.extracted_path and os.path.exists(self.extracted_path):
            print("extracting from volume...")
            shutil.copytree(self.extracted_path, dr)
        else:
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

        # lines = [line for line in lines if "page0" in line]
        # print(lines)
        main_pdf_file = os.path.join(dr, "main.pdf")

        # todo: move this to ZIP
        pdfs = unpack_pdf(lines, dr, main_pdf_file)

        return dr, lines, pdfs, file_paths

    @enter()
    def on_enter(self):
        zip_path = os.path.join(VOLUME_PATH, self.args.paperid, f"{self.args.paperid}.zip")
        if os.path.exists(zip_path):
            self.dr, self.lines, self.pdfs, self.file_paths = self.inititalize_directory(self.args)

        # print(self.extracted_path, os.path.exists(self.extracted_path))
        # if self.extracted_path and os.path.exists(self.extracted_path):
        #    self.dr, self.lines, self.pdfs, self.file_paths = self.inititalize_directory(self.args)

    @method()
    def download_and_zip(self, paperid: str):
        import shutil
        volume_path = f"{VOLUME_PATH}/{paperid}/{paperid}.zip"
        zip_file, abstract, title = self.get_zip()

        # copy zip_file to shared volume
        os.makedirs(os.path.dirname(volume_path), exist_ok=True)
        shutil.copy(zip_file, volume_path)

        volume.commit()

        return volume_path, abstract, title

    def get_speech_blocks_for_video(self, text, pageblockmap):
        splits = sent_tokenize(text)

        assert len(pageblockmap) == len(splits), f"pageblock splits mismatch {len(pageblockmap)} :  {len(splits)}"

        block_text = []
        prev = pageblockmap[0]
        last_page = 0

        for ind, m in enumerate(pageblockmap):
            if m == prev:
                block_text.append(splits[ind])
                continue

            joinedtext = " ".join(block_text)
            if isinstance(prev, list):
                last_page = prev[1]
                yield joinedtext, prev[1], prev[2]
            else:
                yield joinedtext, last_page, None

            prev = m
            block_text = [splits[ind]]

    def get_mp3_name(self, text, page=None, block=None):
        h = hash(text)
        if page is None and block is None:
            return f"short_{h}.mp3"
        elif block is None:
            return f"page{page}summary_{h}.mp3"
        else:
            return f"page{page}block{block}_{h}.mp3"

    @method()
    def text_to_speech(self, text, chunk_audio_file_name) -> tuple[bytes, str]:
        processor = DocumentProcessor(self.args)

        audio, file_path = processor.tts_client.synthesize_speech(
            text, self.args.voice, 1.0, ".", chunk_audio_file_name
        )

        assert audio is not None, "Audio is None"
        assert file_path is not None, "File path is None"

        return audio, chunk_audio_file_name, text

    @method()
    def q_a_to_speech(self, text, is_question) -> str:
        processor = DocumentProcessor(self.args)
        voice = "Studio-Q" if is_question else "Studio-A"
        chunk_audio_file_name = f"question_{hash(text)}.mp3" if is_question else f"answer_{hash(text)}.mp3"

        files_dir = "."
        audio_content, file_path = processor.tts_client.synthesize_speech(
            text, voice, 1.0, files_dir, chunk_audio_file_name
        )

        if is_question:
            chunk_audio_with_silence = os.path.join(files_dir, f"with_silence_{hash(text)}.mp3")
            os.system(
                f'{self.args.ffmpeg} -i {file_path} -f lavfi -t 2 -i anullsrc=r=44100:cl=stereo -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1" {chunk_audio_with_silence}'
            )
            shutil.move(chunk_audio_with_silence, file_path)

            with open(file_path, "rb") as f:
                audio_content = f.read()

        return audio_content, chunk_audio_file_name, text, is_question

    @method()
    def verbalize_section(self, sec, pagemap_section):
        verbalizer = Verbalizer(self.args, get_matcher(), logging)
        return verbalizer.process_section(sec, pagemap_section)

    @method()
    def get_gpt_text(self, i, message_batch, model: str, message_type: str) -> tuple[Any, str, str]:
        verbalizer = Verbalizer(self.args, get_matcher(), logging)
        gpt_text, _, _ = verbalizer.get_gpt_text(message_batch, model)
        return i, gpt_text, message_type

    def get_zip(self) -> tuple[str, Any | str, Any | str]:
        processor = DocumentProcessor(self.args)
        args = self.args

        checkpoint = str(Path(VOLUME_PATH) / Path(self.run_id) / Path("checkpoint"))
        checkpoint_zip = checkpoint + "/checkpoint.zip"
        if os.path.exists(checkpoint_zip):
            print("restoring from checkpoint: ", checkpoint)
            with open(os.path.join(checkpoint, "pageblockmap.pkl"), "rb") as f:
                pageblockmap = pickle.load(f)

            with open(os.path.join(checkpoint, "state.pkl"), "rb") as f:
                state = pickle.load(f)

                main_file = state["main_file"]
                files_dir = state["files_dir"]
                title = state["title"]
                text = state["text"]
                abstract = state["abstract"]

            with zipfile.ZipFile(checkpoint_zip, "r") as zipf:
                zipf.extractall(".")

            processor.files_dir = files_dir
            list_files(files_dir)
            print("files restored")

        else:
            print(" ==== LATEX =============================================")
            main_file, files_dir = processor.process_latex()
            print(" ==== HTML ==============================================")
            title, text, abstract = processor.process_html(main_file)
            # if self.args.extract_text_only:
            #    return None
            print(" ==== MAP ================================================")
            _, pageblockmap = self.process_map(main_file, files_dir, text)

            # zip all files in all directories under files_dir
            print(" ==== CHECKPOINT =========================================")

            zip_path = os.path.join(files_dir, "checkpoint.zip")

            walk_zip(files_dir, zip_path)

            os.makedirs(checkpoint, exist_ok=True)
            shutil.copy(zip_path, checkpoint)

            with open(os.path.join(checkpoint, "pageblockmap.pkl"), "wb") as f:
                pickle.dump(pageblockmap, f)

            state = {
                "main_file": main_file,
                "files_dir": files_dir,
                "title": title,
                "text": text,
                "abstract": abstract,
            }

            with open(os.path.join(checkpoint, "state.pkl"), "wb") as f:
                pickle.dump(state, f)

            volume.commit()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:

            futures = []

            @try_decorator
            def create_video() -> dict[str, Any]:
                print(" ==== create_video ==============================================")
                verbalizer = Verbalizer(args, get_matcher(), logging)

                # generate requests for llm
                message_batches = list(verbalizer.generate_messages(text, pageblockmap))

                # build args for the get_gpt_text function
                star_args = [
                    (i, batch, model, t)
                    for i, (batches, _, _, _, _) in enumerate(message_batches)
                    for batch, model, t in batches
                ]

                # process all the gpt requests in parallel
                llm_responses = list(self.get_gpt_text.starmap(star_args))
                llm_responses.sort(key=lambda x: x[0])

                # Group llm_responses by i using itertools
                grouped_responses = itertools.groupby(llm_responses, key=lambda x: x[0])
                # grouped_responses.sort(key=lambda x: x[0])

                matched_responses = []
                for i, responses in grouped_responses:
                    # Process the responses for each group
                    _, sectionName, _, curr_upd, page_inds = message_batches[i]
                    llm_texts = [(t, llm_text) for _, llm_text, t in responses]
                    matched_responses.append((sectionName, curr_upd, page_inds, llm_texts))

                gpttext, gptpagemap, verbalizer_steps, textpagemap = verbalizer.process_results(matched_responses)

                with open(os.path.join(processor.files_dir, "gptpagemap.pkl"), "wb") as f:
                    pickle.dump(gptpagemap, f)

                with open(os.path.join(processor.files_dir, "gpt_verb_steps.txt"), "w") as f:
                    for si, s in enumerate(verbalizer_steps):
                        f.write(f"===Original {si}===\n\n")
                        f.write(s[0])
                        f.write("\n\n")
                        f.write(f"===GPT {si}===\n\n")
                        f.write(s[1])
                        f.write("\n\n")

                with open(os.path.join(processor.files_dir, "gpt_text.txt"), "w") as f:
                    f.write(gpttext)

                logging.info(f"Extracted text:\n\n {gpttext}")

                print("=" * 10, "text to speech", "=" * 30)
                speech = list(self.get_speech_blocks_for_video(gpttext, gptpagemap))

                for t, page, block in speech:
                    # print the first 10 chars of text
                    print(f"text:{t[:10]} page: {page}, block: {block}")

                file_names = [(t, self.get_mp3_name(t, page, block)) for t, page, block in speech]

                results = list(self.text_to_speech.starmap(file_names))
                with open(os.path.join(processor.files_dir, self.args.chunk_mp3_file_list), "w") as mp3_list_file:
                    for audio_data, chunk_audio_file_name, _ in results:
                        mp3_file = os.path.join(processor.files_dir, os.path.basename(chunk_audio_file_name))
                        with open(mp3_file, "wb") as f:
                            f.write(audio_data)

                        print(self.args.chunk_mp3_file_list, mp3_file)
                        mp3_list_file.write(f"file {chunk_audio_file_name}\n")

                return {
                    "gpttext": gpttext,
                    "gptpagemap": gptpagemap,
                    "verbalizer_steps": verbalizer_steps,
                    "textpagemap": textpagemap,
                }

            @try_decorator
            def create_short() -> dict[str, Any]:
                print(" ==== create_short ==============================================")
                gpttext_short, slides_short = gpt_short_verbalizer(
                    processor.files_dir, processor.llm_api, args.llm_strong, args.llm_base, logging
                )
                with open(os.path.join(processor.files_dir, "gpt_text_short.txt"), "w") as f:
                    f.write(gpttext_short)

                with open(os.path.join(processor.files_dir, "gpt_slides_short.json"), "w") as json_file:
                    json.dump(slides_short, json_file, indent=4)

                print(slides_short)

                tts = VerbalizerShort(args, files_dir, logging)
                messages = tts.generate_messages(gpttext_short, slides_short["slides"])

                star_args = [(m, self.get_mp3_name(m, None, None)) for m in messages]

                results = list(self.text_to_speech.starmap(star_args))

                _, final_audio_short = tts.process_files(results, slides_short)

                if self.args.gdrive_id:
                    self.gdrive_client.upload_audio(f"[short] {title}", f"{final_audio_short}")
                    self.logging.info("Uploaded short audio to GDrive")

                return {"gpttext_short": gpttext_short, "gptslides_short": slides_short["slides"]}

            @try_decorator
            def create_qa() -> dict[str, Any]:
                print(" ==== create_qa ==============================================")
                with open(os.path.join(processor.files_dir, "original_text_split_pages.txt")) as f:
                    paper_text = f.read()

                verbalizer = QAVerbalizer(args, get_matcher(), logging)
                messages = list(verbalizer.prepare_messages(paper_text))

                questions = [q for q, _ in messages]

                star_args = [(i, m, self.args.llm_base, "qa") for i, (_, m) in enumerate(messages)]
                results = list(self.get_gpt_text.starmap(star_args))
                results.sort(key=lambda x: x[0])
                answers, qa_pages = verbalizer.process_answers([gpt_text for _, gpt_text, _ in results], paper_text)

                create_questions(questions, os.path.join(processor.files_dir, "questions"))

                with open(os.path.join(processor.files_dir, "qa_pages.pkl"), "wb") as f:
                    pickle.dump(qa_pages, f)

                with open(os.path.join(processor.files_dir, "gpt_questions_answers.txt"), "w") as f:
                    for q, a in zip(questions, answers):
                        f.write(f"==== Question ====\n\n")
                        f.write(q)
                        f.write("\n\n")
                        f.write(f"==== Answer ====\n\n")
                        f.write(a)
                        f.write("\n\n")

                qa_mp3_list_file = os.path.join(processor.files_dir, f"qa_{self.args.chunk_mp3_file_list}")
                with open(qa_mp3_list_file, "w") as mp3_list_file:
                    star_args = [
                        (text, is_question)
                        for q, a in zip(questions, answers)
                        for text, is_question in processor.extract_q_a(q, a)
                    ]

                    results = self.q_a_to_speech.starmap(star_args)

                    for audio, file_name, _, _ in results:
                        mp3_file = os.path.join(processor.files_dir, os.path.basename(file_name))
                        with open(mp3_file, "wb") as f:
                            f.write(audio)
                        print("wrote ", mp3_file)
                        mp3_list_file.write(f"file {file_name}\n")

                final_audio_qa = os.path.join(processor.files_dir, f"{self.args.final_audio_file}_qa.mp3")
                os.system(f"{self.args.ffmpeg} -f concat -i {qa_mp3_list_file} " f"-c copy {final_audio_qa}")

                logging.info("Created QA audio file")

                if self.args.gdrive_id:
                    self.gdrive_client.upload_audio(f"[QA] {title}", f"{final_audio_qa}")
                    logging.info("Uploaded QA audio to GDrive")

                return {"gpttext_q": questions, "gpttext_a": answers, "qa_pages": qa_pages}

            @try_decorator
            def create_simple() -> dict[str, Any]:
                print(" ==== create_simple ==============================================")
                gpttext, verbalizer_steps = gpt_text_verbalizer(
                    text, processor.llm_api, args.llm_base, args.manual_gpt, args.include_summary, logging
                )
                processor.process_simple_speech(gpttext)
                return {"gpttext": gpttext}

            if args.create_short:
                futures.append(executor.submit(create_short))

            if args.create_qa:
                futures.append(executor.submit(create_qa))

            if args.create_video:
                futures.append(executor.submit(create_video))

            if args.create_audio_simple:
                futures.append(executor.submit(create_simple))

            create_summary(
                abstract, title, processor.args.paperid, processor.llm_api, processor.args.llm_base, processor.files_dir
            )

            concurrent.futures.wait(futures)

            tmpdata = {}
            for future in futures:
                result, ex = future.result()
                if not ex:
                    tmpdata.update(result)
                else:
                    raise ex

            if len(tmpdata) > 0:
                with open(os.path.join(processor.files_dir, "tmpdata.pkl"), "wb") as f:
                    pickle.dump(tmpdata, f)

        final_audio = os.path.join(processor.files_dir, f"{processor.args.final_audio_file}.mp3")
        os.system(
            f"{processor.args.ffmpeg} -f concat -i {os.path.join(processor.files_dir, processor.args.chunk_mp3_file_list)} -c copy {final_audio}"
        )
        processor.logging.info("Created audio file")

        if processor.args.gdrive_id:
            processor.gdrive_client.upload_audio(title, f"{final_audio}")
            processor.logging.info("Uploaded audio to GDrive")

        return processor.process_zip(main_file), abstract, title

    def process_map(self, main_file, files_dir, text) -> tuple[None, None] | tuple[list[int], list]:
        if not self.args.create_video:
            return None, None

        matcher = get_matcher()

        logging.info("Mapping text to pages")
        pdf_file = f"{os.path.join(files_dir, main_file)}.pdf"
        print(pdf_file)
        pagemap = map_text_to_pdfpages(text, pdf_file, matcher)

        logging.info("Mapping pages to blocks")

        splits = sent_tokenize(text)
        pageblockmap = []
        coords = []

        # move the pdf to the shared volume so process_page containers can access it
        volume_file = os.path.join(VOLUME_PATH, self.run_id, "main.pdf")
        os.makedirs(os.path.join(VOLUME_PATH, self.run_id), exist_ok=True)
        shutil.copy(pdf_file, volume_file)
        volume.commit()

        star_args = []
        for i, pg_num in enumerate(np.unique(pagemap)):
            np_array = np.array(pagemap)
            start = np.where(np_array == pg_num)[0][0]
            end = np.where(np_array == pg_num)[0][-1] + 1
            page_text_splits = splits[start:end]
            star_args.append((i, pg_num, page_text_splits, volume_file))

        results = list(self.process_page.starmap(star_args))
        results.sort(key=lambda x: x[0])

        print(f"Done processing {len(results)} pages")

        for i, p, smoothed_seq, good_coords in results:
            page_pdf = f"{os.path.join(files_dir, str(p))}.pdf"
            page_png = f"{os.path.join(files_dir, str(p))}.png"

            os.system(
                f"{self.args.gs} -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dFirstPage={p + 1} -dLastPage={p + 1} -sOutputFile={page_pdf} {pdf_file} > /dev/null 2>&1"
            )
            os.system(f"{self.args.gs} -sDEVICE=png16m -r400 -o {page_png} {page_pdf} > /dev/null 2>&1")
            pageblockmap += [[i, p, s] for s in smoothed_seq]
            coords.append(good_coords)

        # self.pickle_dump(pagemap, os.path.join(self.files_dir, "map_pagemap.pkl"))
        # self.pickle_dump(pageblockmap, os.path.join(self.files_dir, "map_pageblockmap.pkl"))

        with open(os.path.join(files_dir, "block_coords.pkl"), "wb") as f:
            pickle.dump(coords, f)

        with open(os.path.join(files_dir, "original_text_split_pages.txt"), "w") as f:
            splits = sent_tokenize(text)

            for i, p in enumerate(np.unique(pagemap)):
                start = np.where(np.array(pagemap) == p)[0][0]
                end = np.where(np.array(pagemap) == p)[0][-1] + 1
                chunk_text = " ".join(splits[start:end])
                f.write(f"PAGE {p + 1}\n\n")
                f.write(chunk_text)
                f.write("\n\n")

        return pagemap, pageblockmap

    @method()
    def process_page(self, i, pg_num, page_text_splits, pdf_file) -> list[int]:
        matcher = get_matcher()
        print(pdf_file, os.path.exists(pdf_file))

        print("------ process_page", i, pg_num)
        files_dir = f".temp/{self.args.paperid}_files"
        os.makedirs(files_dir, exist_ok=True)

        page_pdf = f"{os.path.join(files_dir, str(pg_num))}.pdf"
        page_png = f"{os.path.join(files_dir, str(pg_num))}.png"

        ret = os.system(
            f"{self.args.gs} -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dFirstPage={pg_num + 1} -dLastPage={pg_num + 1} -sOutputFile={page_pdf} {pdf_file} > /dev/null 2>&1"
        )

        if ret != 0:
            raise Exception("Error extracting page")

        doc = fitz.open(page_pdf)
        page = doc[0]

        blocks = page.get_text("blocks")
        good_blocks = []
        good_coords = []
        for b in blocks:
            if len(b[4]) > 0:
                good_blocks.append(b[4])
                good_coords.append(list(b[:4]))

        seq = matcher.match(
            page_text_splits,
            good_blocks,
            bert=True,
            minilm=True,
            fuzz=True,
            spacy=True,
            diff=True,
            tfidf=True,
            pnt=True,
        )

        smoothed_seq = smooth_sequence(seq)

        return i, pg_num, smoothed_seq, good_coords

    @method()
    def process_line_f(self, i, line) -> tuple[int, str, bytes, Exception]:
        print("process_line_f", i, line)

        block_coords = pickle.load(open(os.path.join(self.dr, "block_coords.pkl"), "rb"))
        gptpagemap = pickle.load(open(os.path.join(self.dr, "gptpagemap.pkl"), "rb"))

        _, mp4_file, ex = process_line(i, line, self.dr, self.video_args, block_coords, gptpagemap)

        data = b""
        with open(mp4_file, "rb") as f:
            for chunk in f:
                data += chunk

        return i, mp4_file, data, ex

    @method()
    def process_short_line_f(self, i, line, page_num) -> tuple[int, str, bytes, None]:
        _, mp4_file, _, ex = process_short_line(i, line, page_num, self.dr, self.video_args)

        data = b""
        with open(mp4_file, "rb") as f:
            for chunk in f:
                data += chunk

        return i, mp4_file, data, ex

    @method()
    def process_qa_line_f(self, line, line_num, input_path, dr, args) -> tuple[int, str, bytes, Exception]:
        print("process_qa_line_f", line_num, line)
        _, mp4_file, _, ex = process_qa_line(line, line_num, input_path, self.dr, self.video_args)

        data = b""
        with open(mp4_file, "rb") as f:
            for chunk in f:
                data += chunk

        return line_num, mp4_file, data, ex

    @method()
    def makevideo(self, video_type: str):
        pprint(self.lines)
        if video_type == "long":
            results = list(
                self.process_line_f.starmap(
                    [(i, line) for i, line in enumerate(self.lines)],
                    return_exceptions=True,
                )
            )

        elif video_type == "short" and "short_mp3s" in self.file_paths:
            with open(self.file_paths["short_mp3s"], "r") as f:
                lines = f.readlines()

            results = list(self.process_short_line_f.starmap([(i, line, i) for i, line in enumerate(lines)]))

        elif video_type == "qa" and "qa_mp3_list" in self.file_paths:
            with open(self.file_paths["qa_mp3_list"], "r") as f:
                lines = f.readlines()

            # lines = [lines[0], lines[1]]

            qa_pages = pickle.load(open(os.path.join(self.dr, "qa_pages.pkl"), "rb"))
            tasks = prepare_qa_tasks(self.dr, lines, qa_pages, self.video_args)
            results = list(self.process_qa_line_f.starmap(tasks))

        results.sort(key=lambda x: x[0])

        mp4_list = Path(self.dr) / "mp4_list.txt"
        with open(mp4_list, "w") as mp4f:
            for _, mp4_file, data, ex in results:
                if ex:
                    logging.error(f"Error occurred: {ex}")
                    return
                else:
                    with open(mp4_file, "wb") as f:
                        f.write(data)
                    print(mp4_file)
                    mp4f.write(f"file {os.path.basename(mp4_file)}\n")
        import subprocess

        command, mp4_output = self.create_video_command(self.args, mp4_list, Path(self.dr) / f"output_{video_type}.mp4")
        print(command)
        proc_result = subprocess.run(command, shell=True)

        # walk_zip(self.dr, f"{self.args.paperid}_long_video.zip")

        # shutil.copy(
        #     f"{self.args.paperid}_long_video.zip",
        #     f"{VOLUME_PATH}/{self.args.paperid}_long_video.zip",
        # )

        # volume.commit()

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
    def run_zip(self, paperid, video_types=["long"]) -> tuple[dict[str, bytes], bytes]:
        zip_file, abstract, title = self.download_and_zip.remote(paperid)
        print(title)
        print(abstract)
        volume.reload()
        with open(zip_file, "rb") as f:
            zip_data = f.read()

        videos = {}

        # volume.delete_file(zip_file)
        return videos, zip_data

    @method()
    def run(self, paperid, video_types=["long"]) -> tuple[dict[str, bytes], bytes]:
        zip_file, abstract, title = self.download_and_zip.remote(paperid)
        print(title)
        print(abstract)
        volume.reload()
        with open(zip_file, "rb") as f:
            zip_data = f.read()

        results = list(self.makevideo.map(video_types))

        videos = {}
        for data, video_type in results:
            videos[video_type] = data

        # volume.delete_file(zip_file)
        return videos, zip_data, abstract, title

    @method()
    def make_video(self, paperid, video_types=["long"]) -> tuple[dict[str, bytes], bytes]:

        # self.extracted_path = os.path.join(VOLUME_PATH, self.run_id, "extracted")
        # os.makedirs("/root/shared/20240501011049715413/extracted", exist_ok=True)
        # print(self.extracted_path)
        # self.inititalize_directory(self.args, target_dir="/root/shared/20240501011049715413/extracted")
        # volume.commit()

        # return None, None

        volume.reload()

        results = list(self.makevideo.map(video_types))

        videos = {}
        for data, video_type in results:
            videos[video_type] = data

        zip_file = f"{VOLUME_PATH}/{paperid}/{paperid}.zip"
        zip_data = open(zip_file, "rb").read()
        # volume.delete_file(zip_file)
        return videos, zip_data


# @stub.function(image=app_image)
# @web_endpoint()
# def get_storage(key: str):
#     return timecop_storage.get(key) if timecop_storage.contains(key) else {}


@app.local_entrypoint()
def main():

    paperid = "1706.03762"
    video_types = [
        "long",
        # "short",
        # "qa",
    ]

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
        gdrive_id=None,
        voice="onyx",
        final_audio_file="final_audio",
        chunk_mp3_file_list="mp3_list.txt",
        manual_gpt=False,
        include_summary=True,
        extract_text_only=False,
        create_video="long" in video_types,
        create_short="short" in video_types,
        create_qa="qa" in video_types,
        create_audio_simple=False,
        llm_strong="gpt-4-0125-preview",
        llm_base="gpt-3.5-turbo-0125",  # "gpt-4-0125-preview",
        openai_key=os.environ.get("OPENAI_API_KEY", ""),
        tts_client="openai",
    )

    video_args = argparse.Namespace(paperid=paperid, gs="gs", ffmpeg="ffmpeg", ffprobe="ffprobe")

    arx = ArxivVideo(args, video_args)

    # images, zip, abstract, title = arx.run.remote(paperid, video_types)
    # print(title)
    # print(abstract)
    images, zip = arx.run_zip.remote(paperid, video_types)

    # images, zip = arx.make_video.remote(paperid, video_types)

    if not zip:
        return

    zip_file = f".temp/{paperid}/{paperid}.zip"
    os.makedirs(os.path.dirname(zip_file), exist_ok=True)
    with open(zip_file, "wb") as f:
        f.write(zip)
        print(f"Saved {zip_file}")

    if not images:
        return

    for k, v in images.items():
        mp4_file = f".temp/{paperid}/{paperid}_{k}.mp4"
        with open(mp4_file, "wb") as f:
            f.write(v)
            print(f"Saved {mp4_file}")


def test_args():
    paperid = "1706.03762"
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
    return args


def test():
    args = test_args()
    v = VerbalizerShort(args, ".temp", logging)
    print(v.files_dir)

    video_args = argparse.Namespace(paperid=args.paperid, gs="gs", ffmpeg="ffmpeg", ffprobe="ffprobe")

    ax = ArxivVideo(args, video_args)

    with open(f".temp/{args.paperid}_files/gpt_text.txt", "r") as f:
        tx = f.read()

    with open(f".temp/{args.paperid}_files/gptpagemap.pkl", "rb") as f:
        pagemap = pickle.load(f)

    speech = list(ax.get_speech_blocks_for_video(tx, pagemap))
    for t, page, block in speech:
        # print the first 10 chars of text
        print(f"text:{t[:10]} page: {page}, block: {block}")


def test2():
    with open(".temp/1706.03762_files/extracted_orig_text_clean.txt", "r") as f:
        text = f.read()

    with open(".temp/1706.03762_files/map_pageblockmap.pkl", "rb") as f:
        pageblockmap = pickle.load(f)

    from unittest.mock import Mock
    import logging
    from map.utils import Matcher
    from pprint import pprint
    from nltk.tokenize import sent_tokenize

    gpttext, gptpagemap, verbalizer_steps, textpagemap = gpt_textvideo_verbalizer(
        text,
        mock_llm_api,  # openai.chat.completions.create,
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-0125",
        False,
        True,
        pageblockmap,
        Matcher(cache_dir="cache"),
        logging,
    )

    pprint(textpagemap)
    pprint(gptpagemap)
    print(len(gptpagemap))
    print(len(sent_tokenize(gpttext)))


def img_test():

    dr = "/workspaces/ArxivPapers/.temp/1706.03762"
    video_args = argparse.Namespace(paperid="1706.03762", gs="gs", ffmpeg="ffmpeg", ffprobe="ffprobe")
    block_coords = pickle.load(
        open("/workspaces/ArxivPapers/.temp/debug/long_video/.temp/1706.03762/output/block_coords.pkl", "rb")
    )
    gptpagemap = pickle.load(
        open("/workspaces/ArxivPapers/.temp/debug/long_video/.temp/1706.03762/output/gptpagemap.pkl", "rb")
    )

    with open("/workspaces/ArxivPapers/.temp/debug/long_video/.temp/1706.03762/output/mp3_list.txt", "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):

        process_line(
            i,
            line,
            "/workspaces/ArxivPapers/.temp/debug/long_video/.temp/1706.03762/output/",
            video_args,
            block_coords,
            gptpagemap,
        )


def restore():
    checkpoint_zip = "/workspaces/ArxivPapers/checkpoint.zip"
    if os.path.exists(checkpoint_zip):
        with open("/workspaces/ArxivPapers/pageblockmap.pkl", "rb") as f:
            pageblockmap = pickle.load(f)

        with open("/workspaces/ArxivPapers/state.pkl", "rb") as f:
            state = pickle.load(f)

            main_file = state["main_file"]
            files_dir = state["files_dir"]
            title = state["title"]
            text = state["text"]
            abstract = state["abstract"]

        with zipfile.ZipFile(checkpoint_zip, "r") as zipf:
            zipf.extractall(".")
        processor = DocumentProcessor(test_args())
        processor.files_dir = files_dir

        pprint(pageblockmap)

        with open(".temp/1706.03762_files/map_pageblockmap.pkl", "rb") as f:
            pageblockmap2 = pickle.load(f)

        pprint(pageblockmap2)

        assert pageblockmap == pageblockmap2


# restore()
# test()
# test2()
# img_test()
