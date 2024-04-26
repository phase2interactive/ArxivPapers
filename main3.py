import argparse
import json
import logging
import os
import pickle
import random
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import openai
from nltk.tokenize import sent_tokenize

from tex.utils import *
from htmls.utils import *
from map.utils import *
from gpt.utils import *
from speech.utils import *
from zip.utils import *
from gdrive.utils import *
import random


class DocumentProcessor:
    def __init__(self, args):
        self.args = args
        self.files_dir = None
        self.tts_client = None
        self.logging = None
        self.gdrive_client = None

        self.setup_logging(args)

        if self.args.tts_client == "openai":
            self.tts_client = OpenAITTSClient()
        else:
            self.tts_client = GoogleTTSClient()

        if self.args.gdrive_id:
            self.gdrive_client = GDrive(self.args.gdrive_id)

        openai.api_key = self.args.openai_key
        self.llm_api = openai.chat.completions.create

    def setup_logging(self, args):
        if args.verbose == "debug":
            level = logging.DEBUG
        elif args.verbose == "info":
            level = logging.INFO
        else:
            level = logging.WARNING

        logging.basicConfig(level=level, format="\n%(asctime)s - %(levelname)s - %(message)s")
        self.logging = logging

    def process_latex(self):
        paper_id = self.args.paperid
        remove_oldfiles_samepaper(paper_id)
        self.files_dir = download_paper(paper_id)
        tex_files = get_tex_files(self.files_dir)
        self.logging.info(f"Found .tex files: {tex_files}")
        main_file = get_main_source_file(tex_files, self.files_dir)

        if main_file:
            self.logging.info(f"Found [{main_file}.tex] as a main source file")
        else:
            self.logging.info("Failed to find main source file. Enter manually:")
            main_file = input()

        return main_file

    def process_html(self, main_file):
        html_parser = "latex2html" if self.args.l2h else "latexmlc"
        display = "> /dev/null 2>&1" if self.args.verbose != "debug" else ""

        generate_html(
            main_file,
            self.files_dir,
            html_parser,
            self.args.pdflatex,
            self.args.latex2html,
            self.args.latexmlc,
            self.args.gs,
            display,
            self.logging,
        )

        citations = parse_aux_file(os.path.join(self.files_dir, f"{main_file}.aux"))
        title, content, html, abstract = parse_html_file(
            os.path.join(self.files_dir, main_file, f"{main_file}.html"), html_parser
        )

        self.logging.info(f"Parsed HTML file. Title: {title}")

        stop_words = ["references", "appendix", "conclusion", "acknowledgments", "about this document"]
        if self.args.stop_word:
            stop_words.append(self.args.stop_word)

        self.logging.info(f"Section title stop words: {stop_words}")

        D = create_nested_dict(content, self.logging, stop_words)
        extract_text_recursive(D, self.files_dir, main_file, citations, html_parser, html)
        text = depth_first_search(D)

        splits = sent_tokenize(text)
        text = " ".join(splits)

        with open(os.path.join(self.files_dir, "extracted_orig_text_clean.txt"), "w") as f:
            f.write(text)

        with open(os.path.join(self.files_dir, "original_text_split_sections.txt"), "w") as f:
            sections = text.split(" Section: ")
            for i, s in enumerate(sections):
                f.write(f"SECTION {i + 1}\n\n")
                f.write(s)
                f.write("\n\n")

        return title, text, abstract

    def process_map(self, main_file, text):
        if not self.args.create_video:
            return None, None

        matcher = Matcher(self.args.cache_dir)

        self.logging.info("Mapping text to pages")
        pagemap = map_text_to_pdfpages(text, f"{os.path.join(self.files_dir, main_file)}.pdf", matcher)

        self.logging.info("Mapping pages to blocks")
        coords, pageblockmap = map_page_to_blocks(
            pagemap,
            text,
            self.args.gs,
            self.files_dir,
            f"{os.path.join(self.files_dir, main_file)}.pdf",
            matcher,
            "> /dev/null 2>&1",
        )

        with open(os.path.join(self.files_dir, "block_coords.pkl"), "wb") as f:
            pickle.dump(coords, f)

        with open(os.path.join(self.files_dir, "original_text_split_pages.txt"), "w") as f:
            splits = sent_tokenize(text)

            for i, p in enumerate(np.unique(pagemap)):
                start = np.where(np.array(pagemap) == p)[0][0]
                end = np.where(np.array(pagemap) == p)[0][-1] + 1
                chunk_text = " ".join(splits[start:end])
                f.write(f"PAGE {p + 1}\n\n")
                f.write(chunk_text)
                f.write("\n\n")

        return pagemap, pageblockmap

    def process_gpt(self, text, pagemap, pageblockmap, title, matcher):
        tmpdata = {}

        if self.args.create_short:
            gpttext_short, slides_short = gpt_short_verbalizer(
                self.files_dir, self.llm_api, self.args.llm_strong, self.args.llm_base, self.logging
            )
            with open(os.path.join(self.files_dir, "gpt_text_short.txt"), "w") as f:
                f.write(gpttext_short)

            with open("gpt_slides_short.json", "w") as json_file:
                json.dump(slides_short, json_file, indent=4)

            tmpdata["gpttext_short"] = gpttext_short
            tmpdata["gptslides_short"] = slides_short["slides"]

        if self.args.create_qa:
            questions, answers, qa_pages = gpt_qa_verbalizer(
                self.files_dir, self.llm_api, self.args.llm_base, matcher, self.logging
            )

            create_questions(questions, os.path.join(self.files_dir, "questions"))

            with open(os.path.join(self.files_dir, "qa_pages.pkl"), "wb") as f:
                pickle.dump(qa_pages, f)

            with open(os.path.join(self.files_dir, "gpt_questions_answers.txt"), "w") as f:
                for q, a in zip(questions, answers):
                    f.write("==== Question ====\n\n")
                    f.write(q)
                    f.write("\n\n")
                    f.write("==== Answer ====\n\n")
                    f.write(a)
                    f.write("\n\n")

            tmpdata["gpttext_q"] = questions
            tmpdata["gpttext_a"] = answers
            tmpdata["qa_pages"] = qa_pages

        if self.args.create_video:
            gpttext, gptpagemap, verbalizer_steps, textpagemap = gpt_textvideo_verbalizer(
                text,
                self.llm_api,
                self.args.llm_strong,
                self.args.llm_base,
                self.args.manual_gpt,
                self.args.include_summary,
                pageblockmap,
                matcher,
                self.logging,
            )

            with open(os.path.join(self.files_dir, "gptpagemap.pkl"), "wb") as f:
                pickle.dump(gptpagemap, f)

            tmpdata.update(
                {
                    "gpttext": gpttext,
                    "gptpagemap": gptpagemap,
                    "verbalizer_steps": verbalizer_steps,
                    "textpagemap": textpagemap,
                }
            )

            with open(os.path.join(self.files_dir, "gpt_verb_steps.txt"), "w") as f:
                for si, s in enumerate(verbalizer_steps):
                    f.write(f"===Original {si}===\n\n")
                    f.write(s[0])
                    f.write("\n\n")
                    f.write(f"===GPT {si}===\n\n")
                    f.write(s[1])
                    f.write("\n\n")

            with open(os.path.join(self.files_dir, "gpt_text.txt"), "w") as f:
                f.write(gpttext)

            self.logging.info(f"Extracted text:\n\n {gpttext}")

        if self.args.create_audio_simple:
            gpttext, verbalizer_steps = gpt_text_verbalizer(
                text, self.llm_api, self.args.llm_base, self.args.manual_gpt, self.args.include_summary, self.logging
            )

        if len(tmpdata) > 0:
            with open(os.path.join(self.files_dir, "tmpdata.pkl"), "wb") as f:
                pickle.dump(tmpdata, f)

        return gpttext, gptpagemap if self.args.create_video else None

    def process_speech(self, gpttext, gpttext_short, slides_short, questions, answers, gptpagemap, title):
        if self.args.create_short:
            self.process_short_speech(gpttext_short, slides_short, title)

        if self.args.create_qa:
            self.process_qa_speech(questions, answers, title)

        if self.args.create_video:
            self.process_video_speech(gpttext, gptpagemap)

        if self.args.create_audio_simple:
            self.process_simple_speech(gpttext)

    def process_short_speech(self, gpttext_short, slides_short, title):
        with open(os.path.join(self.files_dir, self.args.chunk_mp3_file_list), "w") as mp3_list_file:
            text_to_speech_short(
                gpttext_short, slides_short, mp3_list_file, self.files_dir, self.tts_client, self.logging
            )

        shutil.copy(
            os.path.join(self.files_dir, self.args.chunk_mp3_file_list),
            os.path.join(self.files_dir, f"shorts_{self.args.chunk_mp3_file_list}"),
        )

        final_audio_short = os.path.join(self.files_dir, f"{self.args.final_audio_file}_short.mp3")
        os.system(
            f"{self.args.ffmpeg} -f concat -i {os.path.join(self.files_dir, self.args.chunk_mp3_file_list)} "
            f"-c copy {final_audio_short} > /dev/null 2>&1"
        )

        self.logging.info("Created short audio file")

        if self.gdrive_client:
            self.gdrive_client.upload_audio(f"[short] {title}", f"{final_audio_short}")
            self.logging.info("Uploaded short audio to GDrive")

        create_slides(slides_short, os.path.join(self.files_dir, "slides"))

    def process_qa_speech(self, questions, answers, title):
        with open(os.path.join(self.files_dir, self.args.chunk_mp3_file_list), "w") as mp3_list_file:
            text_to_speech_qa(
                questions, answers, mp3_list_file, self.files_dir, self.tts_client, self.args.ffmpeg, self.logging
            )

        shutil.copy(
            os.path.join(self.files_dir, self.args.chunk_mp3_file_list),
            os.path.join(self.files_dir, f"qa_{self.args.chunk_mp3_file_list}"),
        )

        final_audio_qa = os.path.join(self.files_dir, f"{self.args.final_audio_file}_qa.mp3")
        os.system(
            f"{self.args.ffmpeg} -f concat -i {os.path.join(self.files_dir, self.args.chunk_mp3_file_list)} "
            f"-c copy {final_audio_qa} > /dev/null 2>&1"
        )

        self.logging.info("Created QA audio file")

        if self.args.gdrive_id:
            self.gdrive_client.upload_audio(f"[QA] {title}", f"{final_audio_qa}")
            self.logging.info("Uploaded QA audio to GDrive")

    def process_video_speech(self, gpttext, gptpagemap):
        with open(os.path.join(self.files_dir, self.args.chunk_mp3_file_list), "w") as mp3_list_file:
            text_to_speechvideo(
                gpttext, mp3_list_file, self.files_dir, self.tts_client, gptpagemap, self.args.voice, self.logging
            )

    def process_simple_speech(self, gpttext):
        with open(os.path.join(self.files_dir, self.args.chunk_mp3_file_list), "w") as mp3_list_file:
            text_to_speech(gpttext, mp3_list_file, self.files_dir, self.tts_client, self.args.voice, self.logging)

    def process_zip(self, main_file):
        renamed_main = "main"
        if main_file != renamed_main:
            temp_filename = f"temp_{random.randint(1000, 9999)}"
            shutil.copy(
                f"{os.path.join(self.files_dir, main_file)}.pdf", f"{os.path.join(self.files_dir, temp_filename)}.pdf"
            )
            shutil.copy(
                f"{os.path.join(self.files_dir, temp_filename)}.pdf",
                f"{os.path.join(self.files_dir, renamed_main)}.pdf",
            )

        crop_pdf(
            f"{self.files_dir}/{main_file}.pdf",
            f"{self.files_dir}/fpage.pdf",
            self.args.gs,
            upper_top=3,
            top_percent=25,
            left_percent=12,
            right_percent=7,
        )

        return zip_files(
            self.files_dir,
            self.args.gs,
            self.args.ffmpeg,
            self.args.create_short,
            self.args.create_qa,
            self.args.create_video,
            self.args.final_audio_file,
            self.args.chunk_mp3_file_list,
            "> /dev/null 2>&1",
        )

    def main(self):
        main_file = self.process_latex()
        title, text, abstract = self.process_html(main_file)
        # if self.args.extract_text_only:
        #    return None

        pagemap, pageblockmap = self.process_map(main_file, text)
        matcher = Matcher(self.args.cache_dir)

        gpttext, gptpagemap = self.process_gpt(text, pagemap, pageblockmap, title, matcher)

        if self.args.create_short:
            gpttext_short, slides_short = gpt_short_verbalizer(
                self.files_dir, self.llm_api, self.args.llm_strong, self.args.llm_base, self.logging
            )
        else:
            gpttext_short, slides_short = None, None

        if self.args.create_qa:
            questions, answers, _ = gpt_qa_verbalizer(
                self.files_dir, self.llm_api, self.args.llm_base, matcher, self.logging
            )
        else:
            questions, answers = None, None

        self.process_speech(gpttext, gpttext_short, slides_short, questions, answers, gptpagemap, title)
        create_summary(abstract, title, self.args.paperid, self.llm_api, self.args.llm_base, self.files_dir)
        final_audio = os.path.join(self.files_dir, f"{self.args.final_audio_file}.mp3")
        os.system(
            f"{self.args.ffmpeg} -f concat -i {os.path.join(self.files_dir, self.args.chunk_mp3_file_list)} -c copy {final_audio} > /dev/null 2>&1"
        )
        self.logging.info("Created audio file")

        if self.args.gdrive_id:
            self.gdrive_client.upload_audio(title, f"{final_audio}")
            self.logging.info("Uploaded audio to GDrive")

        return self.process_zip(main_file)


args = argparse.Namespace(
    paperid="2312.05688",
    l2h=True,
    verbose="debug",
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
    create_qa=False,
    create_audio_simple=False,
    llm_strong="gpt-4-0125-preview",
    llm_base="gpt-4-0125-preview",
    openai_key=os.environ.get("OPENAI_API_KEY", ""),
    tts_client="openai",
)

doc_processor = DocumentProcessor(args)
doc_processor.main()
