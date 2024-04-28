import argparse
import json
import logging
import os
import pickle
import random
import shutil
import concurrent.futures
from typing import Any

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


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")


# create a try decorator
def try_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs), None
        except Exception as e:
            print(f"Error: {e}")
            return None, e

    return wrapper


class Verbalizer:

    def __init__(self, args, matcher, logging):
        self.llm_strong = args.llm_strong
        self.llm_base = args.llm_base
        self.manual = args.manual_gpt
        self.include_summary = args.include_summary
        self.matcher = matcher
        self.logging = logging
        self.encoding = tiktoken.get_encoding("cl100k_base")
        openai.api_key = args.openai_key
        self.llm_api = openai.chat.completions.create

    def get_sections(self, text, pageblockmap):
        splits = sent_tokenize(text)
        sections_split = []
        pagemap_sections = []
        for index, t in enumerate(splits):
            if "Section: ###" in t:
                sections_split.append([t.split("Section: ")[1]])
                pagemap_sections.append([pageblockmap[index]])
            else:
                sections_split[-1].append(t)
                pagemap_sections[-1].append(pageblockmap[index])

        sections = [" ".join(sec) for sec in sections_split]
        return sections, pagemap_sections

    def process_sections(self, sections, pagemap_sections):
        results = [self.process_section(sec, pagemap_sections[i]) for i, sec in enumerate(sections)]
        return self.process_results(results)

    def process_results(self, results):
        gpttext_all = ""
        gptpagemap = []
        textpagemap = []
        verbalizer_steps = []
        for result in results:
            gpttext_all += result["addedtext"] + result["smry"]
            gptpagemap += result["gptpagemap_section"] + result["smry_fakepagemap_section"]
            textpagemap += result["textpagemap_section"] + [-1]
            verbalizer_steps.append(result["verbalizer_step"])

            # Sanity check after processing each section
            if len(sent_tokenize(gpttext_all)) != len(gptpagemap):
                raise Exception("Something went wrong. Mismatch between map and text after processing a section")

        return gpttext_all, gptpagemap, verbalizer_steps, textpagemap

    def process_section(self, sec, pagemap_section) -> dict[str, Any]:
        cleaned_sent_tok = sent_tokenize(sec)
        page_inds = pagemap_section[: len(cleaned_sent_tok)]
        curr_upd = [c.replace("###.", " ").replace("###", "") for c in cleaned_sent_tok]
        curr_upto_upto4k = " ".join(curr_upd)
        while len(self.encoding.encode(curr_upto_upto4k)) > 3800:
            curr_upto_upto4k = curr_upto_upto4k[:-100]

        messages, sectionName = self.prepare_messages(curr_upto_upto4k)
        self.logging.info(f"Section: {sectionName}\n")
        self.logging.info(messages[0]["content"] + " " + messages[1]["content"])
        self.logging.info("-" * 100)

        gpttext, _, _ = self.get_gpt_text(messages)
        addedtext, gptpagemap_section, textpagemap_section = self.finalize_text(
            gpttext, curr_upd, page_inds, sectionName
        )

        smry, smry_fakepagemap_section = self.generate_summary(curr_upto_upto4k) if self.include_summary else ("", [])

        return {
            "addedtext": addedtext,
            "smry": smry,
            "gptpagemap_section": gptpagemap_section,
            "smry_fakepagemap_section": smry_fakepagemap_section,
            "textpagemap_section": textpagemap_section,
            "verbalizer_step": [" ".join(curr_upd), gpttext],
        }

    def prepare_messages(self, curr_upto_upto4k):
        sys_message_func = "You are an ArXiv paper audio paraphraser..."
        human_message = f"The text must be written in the first person plural point of view... <<{curr_upto_upto4k}>>."
        sectionName = self.extract_section_name(curr_upto_upto4k)
        return [
            {"role": "system", "content": sys_message_func},
            {"role": "user", "content": human_message},
        ], sectionName

    def extract_section_name(self, text):
        result = re.search(r"###(.*?)###", text)
        if result:
            sectionName = result.group(1).strip().rstrip(".")
        else:
            sectionName = ""
        return sectionName

    def get_gpt_text(self, messages):
        for i in range(3):
            try:
                response = self.llm_api(model=self.llm_strong, messages=messages, temperature=0)
                gpttext = response.choices[0].message.content
                num_input_tokens = response.usage.prompt_tokens
                num_output_tokens = response.usage.completion_tokens
                break
            except:
                time.sleep(5)
        else:
            raise Exception(f"{self.llm_strong} failed")

        return gpttext, num_input_tokens, num_output_tokens

    def finalize_text(self, gpttext, curr_upd, page_inds, sectionName):
        gpttext = self.clean_text(gpttext)
        addedtext = f" Section: {sectionName}. " + gpttext + " "
        gptpagemap_section, textpagemap_section = map_gpttext_to_text(addedtext, curr_upd, page_inds, self.matcher)
        return addedtext, gptpagemap_section, textpagemap_section

    def clean_text(self, text):
        text = text.replace("$", "").replace("```", "").replace("<<", "").replace(">>", "").replace("**", "")
        text = re.sub(r"\b\w*__\w*\b", "", text)
        text = re.sub("\n+", "\n", text)
        return text

    def generate_summary(self, curr_upto_upto4k):
        sys_message = "As an AI specializing in summarizing ArXiv paper sections..."
        human_message = f"Below is the section that needs to be summarized... <<{curr_upto_upto4k}>>."
        messages = [{"role": "system", "content": sys_message}, {"role": "user", "content": human_message}]
        for i in range(3):
            try:
                response = self.llm_api(model=self.llm_base, messages=messages, temperature=0)
                summary = response.choices[0].message.content
                break
            except:
                time.sleep(5)
        else:
            raise Exception(f"{self.llm_base} failed")
        summary = self.clean_text(summary)
        smry = " Section Summary: " + summary + " "
        smry_fakepagemap_section = [-1] * len(sent_tokenize(smry))
        return smry, smry_fakepagemap_section


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

    def setup_logging(self, args) -> None:
        if args.verbose == "debug":
            level = logging.DEBUG
        elif args.verbose == "info":
            level = logging.INFO
        else:
            level = logging.WARNING

        logging.basicConfig(level=level, format="\n%(asctime)s - %(levelname)s - %(message)s")
        self.logging = logging

    def process_latex(self) -> str:
        paper_id = self.args.paperid
        remove_oldfiles_samepaper(paper_id)
        self.files_dir = download_paper(paper_id)
        tex_files = get_tex_files(self.files_dir)
        self.logging.info(f"Found .tex files: {tex_files}")
        main_file = get_main_source_file(tex_files, self.files_dir)

        if main_file:
            self.logging.info(f"Found [{main_file}.tex] as a main source file")
        else:
            raise Exception("No main source file found")

        return main_file

    def process_html(self, main_file) -> tuple[str, str, str]:
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

    def process_map(self, main_file, text) -> tuple[None, None] | tuple[list[int], list]:
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

    def process_gpt(self, text, pagemap, pageblockmap, title, matcher) -> tuple[str, list | None]:
        tmpdata = {}
        print(" --- process_gpt --- ")

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
            verbalizer = Verbalizer(self.args, matcher, self.logging)
            gpttext, gptpagemap, verbalizer_steps, textpagemap = verbalizer.gpt_textvideo_verbalizer(text, pageblockmap)

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

    def process_short_speech(self, gpttext_short, slides_short, title) -> str:
        print(" --- process_short_speech --- ")
        short_mp3_list_file = os.path.join(self.files_dir, f"shorts_{self.args.chunk_mp3_file_list}")
        with open(short_mp3_list_file, "w") as mp3_list_file:
            text_to_speech_short(
                gpttext_short, slides_short, mp3_list_file, self.files_dir, self.tts_client, self.logging
            )

        final_audio_short = os.path.join(self.files_dir, f"{self.args.final_audio_file}_short.mp3")
        os.system(
            f"{self.args.ffmpeg} -f concat -i {short_mp3_list_file} " f"-c copy {final_audio_short} > /dev/null 2>&1"
        )

        self.logging.info("Created short audio file")

        if self.gdrive_client:
            self.gdrive_client.upload_audio(f"[short] {title}", f"{final_audio_short}")
            self.logging.info("Uploaded short audio to GDrive")

        create_slides(slides_short, os.path.join(self.files_dir, "slides"))

        return short_mp3_list_file

    def extract_q_a(self, q, a):
        yield q, True
        a_sent = a.split(".")[:-1]
        for a_s in a_sent:
            yield a_s, False

    def q_a_to_speech(self, text, is_question) -> str:
        voice = "Studio-Q" if is_question else "Studio-A"
        chunk_audio_file_name = f"qa_{hash(text)}.mp3" if is_question else f"answer_{hash(text)}.mp3"

        audio_content, file_path = self.tts_client.synthesize_speech(
            text, voice, 1.0, self.files_dir, chunk_audio_file_name
        )

        if is_question:
            chunk_audio_with_silence = os.path.join(self.files_dir, f"with_silence_{hash(text)}.mp3")
            os.system(
                f'{self.args.ffmpeg} -i {file_path} -f lavfi -t 2 -i anullsrc=r=44100:cl=stereo -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1" {chunk_audio_with_silence}'
            )
            shutil.move(chunk_audio_with_silence, file_path)

        return chunk_audio_file_name

    def process_qa_speech(self, questions, answers, title) -> str:
        from itertools import starmap, chain

        qa_mp3_list_file = os.path.join(self.files_dir, f"qa_{self.args.chunk_mp3_file_list}")
        with open(qa_mp3_list_file, "w") as mp3_list_file:
            qa_pairs = chain.from_iterable(self.extract_q_a(q, a) for q, a in zip(questions, answers))
            print(qa_pairs)
            for file_name in starmap(self.q_a_to_speech, qa_pairs):
                mp3_list_file.write(f"file '{file_name}'\n")

        final_audio_qa = os.path.join(self.files_dir, f"{self.args.final_audio_file}_qa.mp3")
        os.system(f"{self.args.ffmpeg} -f concat -i {qa_mp3_list_file} " f"-c copy {final_audio_qa}")

        self.logging.info("Created QA audio file")

        if self.args.gdrive_id:
            self.gdrive_client.upload_audio(f"[QA] {title}", f"{final_audio_qa}")
            self.logging.info("Uploaded QA audio to GDrive")

        return qa_mp3_list_file

    def process_video_speech(self, gpttext, gptpagemap) -> None:
        with open(os.path.join(self.files_dir, self.args.chunk_mp3_file_list), "w") as mp3_list_file:
            text_to_speechvideo(
                gpttext, self.tts_client, self.files_dir, mp3_list_file, gptpagemap, self.args.voice, self.logging
            )
        return os.path.join(self.files_dir, self.args.chunk_mp3_file_list)

    def process_simple_speech(self, gpttext) -> None:
        with open(os.path.join(self.files_dir, self.args.chunk_mp3_file_list), "w") as mp3_list_file:
            text_to_speech(gpttext, mp3_list_file, self.files_dir, self.tts_client, self.args.voice, self.logging)

        return os.path.join(self.files_dir, self.args.chunk_mp3_file_list)

    def process_zip(self, main_file) -> str:
        list_files(self.files_dir)
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

        return zip_files_w_args(self.args, self.files_dir)

    def main_threaded(self) -> str:
        args = self.args
        print(" ==== LATEX =============================================")
        main_file = self.process_latex()
        print(" ==== HTML ==============================================")
        title, text, abstract = self.process_html(main_file)
        # if self.args.extract_text_only:
        #    return None

        print(" ==== MAP ==============================================")
        _, pageblockmap = self.process_map(main_file, text)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:

            futures = []

            @try_decorator
            def create_video() -> dict[str, Any]:
                print(" ==== create_video ==============================================")
                verbalizer = Verbalizer(args, Matcher(args.cache_dir), logging)
                gpttext, gptpagemap, verbalizer_steps, textpagemap = verbalizer.process_sections(
                    text,
                    pageblockmap,
                )

                with open(os.path.join(self.files_dir, "gptpagemap.pkl"), "wb") as f:
                    pickle.dump(gptpagemap, f)

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

                logging.info(f"Extracted text:\n\n {gpttext}")

                self.process_video_speech(gpttext, gptpagemap)

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
                    self.files_dir, self.llm_api, args.llm_strong, args.llm_base, logging
                )
                with open(os.path.join(self.files_dir, "gpt_text_short.txt"), "w") as f:
                    f.write(gpttext_short)

                with open("gpt_slides_short.json", "w") as json_file:
                    json.dump(slides_short, json_file, indent=4)

                self.process_short_speech(gpttext_short, slides_short, title)
                return {"gpttext_short": gpttext_short, "gptslides_short": slides_short["slides"]}

            @try_decorator
            def create_qa() -> dict[str, Any]:
                print(" ==== create_qa ==============================================")
                questions, answers, qa_pages = gpt_qa_verbalizer(
                    self.files_dir, self.llm_api, args.llm_base, Matcher(args.cache_dir), logging
                )

                create_questions(questions, os.path.join(self.files_dir, "questions"))

                with open(os.path.join(self.files_dir, "qa_pages.pkl"), "wb") as f:
                    pickle.dump(qa_pages, f)

                with open(os.path.join(self.files_dir, "gpt_questions_answers.txt"), "w") as f:
                    for q, a in zip(questions, answers):
                        f.write(f"==== Question ====\n\n")
                        f.write(q)
                        f.write("\n\n")
                        f.write(f"==== Answer ====\n\n")
                        f.write(a)
                        f.write("\n\n")

                self.process_qa_speech(questions, answers, title)
                return {"gpttext_q": questions, "gpttext_a": answers, "qa_pages": qa_pages}

            @try_decorator
            def create_simple() -> dict[str, Any]:
                print(" ==== create_simple ==============================================")
                gpttext, verbalizer_steps = gpt_text_verbalizer(
                    text, self.llm_api, args.llm_base, args.manual_gpt, args.include_summary, logging
                )
                self.process_simple_speech(gpttext)
                return {"gpttext": gpttext}

            if args.create_short:
                futures.append(executor.submit(create_short))

            if args.create_qa:
                futures.append(executor.submit(create_qa))

            if args.create_video:
                futures.append(executor.submit(create_video))

            if args.create_audio_simple:
                futures.append(executor.submit(create_simple))

            create_summary(abstract, title, self.args.paperid, self.llm_api, self.args.llm_base, self.files_dir)

            concurrent.futures.wait(futures)

            tmpdata = {}
            for future in futures:
                result, ex = future.result()
                if not ex:
                    tmpdata.update(result)
                else:
                    raise ex

            if len(tmpdata) > 0:
                with open(os.path.join(self.files_dir, "tmpdata.pkl"), "wb") as f:
                    pickle.dump(tmpdata, f)

        final_audio = os.path.join(self.files_dir, f"{self.args.final_audio_file}.mp3")
        os.system(
            f"{self.args.ffmpeg} -f concat -i {os.path.join(self.files_dir, self.args.chunk_mp3_file_list)} -c copy {final_audio} > /dev/null 2>&1"
        )
        self.logging.info("Created audio file")

        if self.args.gdrive_id:
            self.gdrive_client.upload_audio(title, f"{final_audio}")
            self.logging.info("Uploaded audio to GDrive")

        return self.process_zip(main_file)

    def main(self) -> str:
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

        if self.args.create_short:
            self.process_short_speech(gpttext_short, slides_short, title)

        if self.args.create_qa:
            self.process_qa_speech(questions, answers, title)

        if self.args.create_video:
            self.process_video_speech(gpttext, gptpagemap)

        if self.args.create_audio_simple:
            self.process_simple_speech(gpttext)

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
    paperid="1706.03762",
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
    create_qa=False,
    create_audio_simple=False,
    llm_strong="gpt-4-0125-preview",
    llm_base="gpt-4-0125-preview",
    openai_key=os.environ.get("OPENAI_API_KEY", ""),
    tts_client="openai",
)

# doc_processor = DocumentProcessor(args)
# doc_processor.main_threaded()


def test():
    # read text and pageblockmap from file
    with open("verbalizer_text.txt", "r") as f:
        text = f.read()
    with open("verbalizer_pageblockmap.pkl", "rb") as f:
        pageblockmap = pickle.load(f)

    with open("process_sections.pkl", "rb") as f:
        results = pickle.load(f)

    verbalizer = Verbalizer(args, Matcher(args.cache_dir), logging)
    # sections, pagemap_sections = verbalizer.get_sections(text, pageblockmap)
    # print(sections, pagemap_sections)

    # results = [verbalizer.process_section(sec, pagemap_sections[i]) for i, sec in enumerate(sections)]

    # # save results
    # with open("process_sections.pkl", "wb") as f:
    #     pickle.dump(results, f)

    gpttext, gptpagemap, verbalizer_steps, textpagemap = verbalizer.process_results(results)

    for si, s in enumerate(verbalizer_steps):
        print(type(s[0]), type(s[1]))

        print(f"===Original {si}===\n\n")
        print(s[0])
        print("\n\n")
        print(f"===GPT {si}===\n\n")
        print(s[1])
        print("\n\n")


#test()
