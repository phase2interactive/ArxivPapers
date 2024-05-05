import argparse
import json
import logging
import os
import pickle
import random
import shutil
import concurrent.futures
from typing import Any
import random
from pprint import pprint

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


class VerbalizerShort:

    def __init__(self, args, files_dir, logging):
        self.args = args
        self.llm_strong = args.llm_strong
        self.llm_base = args.llm_base
        self.manual = args.manual_gpt
        self.include_summary = args.include_summary
        self.logging = logging
        self.encoding = tiktoken.get_encoding("cl100k_base")
        openai.api_key = args.openai_key
        self.llm_api = openai.chat.completions.create
        self.files_dir = files_dir

    def generate_messages(self, gpttext_short: str, slides):
        print(" --- process_short_speech --- ")
        para = gpttext_short.split("\n\n")

        if len(para) == len(slides):
            first_para_sent = sent_tokenize(para[0])
            para[0] = " ".join(first_para_sent[1:])
            para.insert(0, first_para_sent[0])

        yield from para

    def text_to_speech(self, text):
        audio, file_path = self.tts_client.synthesize_speech(
            text, "Neural2-F", 1.0, self.files_dir, f"short_{hash(text)}.mp3"
        )

        logging.info(f"Processed block text: \n\n {text}")

        if os.path.getsize(file_path) == 0:
            return None

        return audio, file_path, text

    def process_files(self, files, slides_short) -> str:
        print(" --- process_files --- ")
        short_mp3_list_file = os.path.join(self.files_dir, f"shorts_{self.args.chunk_mp3_file_list}")
        final_audio_short = os.path.join(self.files_dir, f"{self.args.final_audio_file}_short.mp3")

        with open(short_mp3_list_file, "w") as mp3_list_file:
            for audio_bytes, chunk_audio, _ in files:
                local_mp3 = os.path.join(self.files_dir, os.path.basename(chunk_audio))
                with open(local_mp3, "wb") as f:
                    written = f.write(audio_bytes)
                    print(f"Wrote {written} bytes to {local_mp3}")

                mp3_list_file.write(f"file {os.path.basename(chunk_audio) }\n")

        ret_val = os.system(f"{self.args.ffmpeg} -f concat -i {short_mp3_list_file} -c copy {final_audio_short}")
        if ret_val != 0:
            logging.error(f"ffmpeg error:  {ret_val}")
            raise Exception("ffmpeg error")
        else:
            logging.info("Created short audio file")

        create_slides(slides_short, os.path.join(self.files_dir, "slides"))

        return short_mp3_list_file, final_audio_short


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

    def generate_messages(self, text, pageblockmap):
        sections, pagemap_secations = self.get_sections(text, pageblockmap)

        yield from self.generate_messages_for_sections(sections, pagemap_secations)

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

    def generate_messages_for_sections(self, sections, pagemap_sections):
        curr_upd = []
        curr = ""
        page_inds = []
        for i_s, sec in enumerate(sections):

            # if there is a mismatch, do this hack to make them of equal length
            if len(sent_tokenize(sec)) != len(pagemap_sections[i_s]):
                minN = min(len(sent_tokenize(sec)), len(pagemap_sections[i_s]))
                cleaned_sent_tok = sent_tokenize(sec)[:minN]
                cleaned_pmap_sec = pagemap_sections[i_s][:minN]
            else:
                cleaned_sent_tok = sent_tokenize(sec)
                cleaned_pmap_sec = pagemap_sections[i_s]

            page_inds += cleaned_pmap_sec

            curr += sec

            for c in cleaned_sent_tok:
                curr_upd.append(c.replace("###.", " ").replace("###", ""))

            if i_s < len(sections) - 1 and len(self.encoding.encode(curr)) < 1000:
                continue

            re_result = re.search(r"###(.*?)###", curr)
            if re_result:
                sectionName = re_result.group(1)
            else:
                sectionName = ""

            sectionName = sectionName.rstrip()
            if sectionName.endswith("."):
                sectionName = sectionName[:-1]

            curr_upto_upto4k = " ".join(curr_upd)
            while len(self.encoding.encode(curr_upto_upto4k)) > 3800:
                curr_upto_upto4k = curr_upto_upto4k[:-100]

            messages = [(self.prepare_messages(curr_upto_upto4k), self.llm_strong, "paraphrase")]

            if self.include_summary:
                messages.append((self.prepare_summary_messages(curr_upto_upto4k), self.llm_base, "summary"))

            yield messages, sectionName, curr_upto_upto4k, curr_upd, page_inds
            page_inds = []
            curr_upd = []
            curr_upto_upto4k = ""
            curr = ""

    def get_gpt_responses(self, section_messages) -> list[tuple[Any, Any, Any, list[tuple[Any, str]]]]:
        return [
            (sectionName, curr_upd, page_inds, [(t, self.get_gpt_text(batch, model)[0]) for batch, model, t in batches])
            for batches, sectionName, curr_upto_upto4k, curr_upd, page_inds in section_messages
        ]

    def process_results(self, responses: tuple[str, Any, Any, list[tuple[Any, str]]]) -> tuple[str, list, list, list]:
        gpttext_all = ""
        gptpagemap = []
        textpagemap = []
        verbalizer_steps = []

        for sectionName, curr_upd, page_inds, gpt_response in responses:
            prefix = {
                "paraphrase": f"Section: {sectionName}.",
                "summary": " Section Summary: ",
            }

            response_by_type = {
                message_type: " ".join([gpt_text for t, gpt_text in gpt_response if message_type == t])
                for message_type in set(t for t, _ in gpt_response)
            }

            # expand response_by_type by converting the value to a tuple
            response_by_type = {k: (v, f"{prefix[k]} {v} ") for k, v in response_by_type.items()}

            gpttext, prefixed_text = response_by_type["paraphrase"]
            gptpagemap_section, textpagemap_section = map_gpttext_to_text(
                prefixed_text, curr_upd, page_inds, self.matcher
            )
            _, prefixed_smry = response_by_type["summary"]
            smry_fakepagemap_section = [-1] * len(sent_tokenize(prefixed_smry))

            gpttext_all += "".join([v[1] for _, v in response_by_type.items()])
            gptpagemap += gptpagemap_section + smry_fakepagemap_section
            textpagemap += textpagemap_section + [-1]

            verbalizer_steps.append([" ".join(curr_upd), gpttext])

            assert len(sent_tokenize(gpttext_all)) == len(gptpagemap), (
                "text and page mapping mismatch in section " + sectionName
            )

        return gpttext_all, gptpagemap, verbalizer_steps, textpagemap

    def prepare_messages(self, curr_upto_upto4k) -> list[dict[str, str]]:
        sys_message_func = (
            f"You are an ArXiv paper audio paraphraser. Your primary goal is to "
            "rephrase the original paper content while preserving its overall "
            "meaning and structure, but simplifying along the way, and make it "
            "easier to understand. In the event that you encounter a "
            "mathematical expression, it is essential that you verbalize it in "
            "straightforward nonlatex terms, while remaining accurate, and "
            "in order to ensure that the reader can grasp the equation's "
            "meaning solely through your verbalization. Do not output any long "
            "latex expressions, summarize them in words."
        )
        human_message = (
            "The text must be written in the first person plural point of view. Do not use long latex "
            "expressions, paraphrase them or summarize in words. Be as faithful to the given text as "
            "possible. Below is the section of the paper, requiring paraphrasing and simplification "
            f' and it is indicated by double angle brackets <<{curr_upto_upto4k}>>. Start with "In this '
            'section, we" and continue in first person plural point of view.'
        )

        return [
            {"role": "system", "content": sys_message_func},
            {"role": "user", "content": human_message},
        ]

    def prepare_summary_messages(self, curr_upto_upto4k) -> list[dict[str, str]]:
        sys_message = (
            "As an AI specializing in summarizing ArXiv paper sections, your main task is to distill complex "
            "scientific concepts from a given section of a paper into 2-3 simple, yet substantial, "
            "sentences. Retain key information, deliver the core idea, and ensure the summary is easy "
            "to understand, while not losing the main essence of the content. "
        )

        human_message = (
            "Below is the section that needs to be summarized in at most 2-3 sentences and it is "
            f'indicated by double angle brackets <<{curr_upto_upto4k}>>. Start with "In this '
            'section, we" and continue in first person plural point of view.'
        )
        messages = [{"role": "system", "content": sys_message}, {"role": "user", "content": human_message}]
        return messages

    def extract_section_name(self, text):
        result = re.search(r"###(.*?)###", text)
        if result:
            sectionName = result.group(1).strip().rstrip(".")
        else:
            sectionName = ""
        return sectionName

    def get_gpt_text(self, messages: list[dict[str, str]], model: str) -> tuple[str, int, int]:
        response = self.llm_api(model=model, messages=messages, temperature=0)
        gpttext = response.choices[0].message.content
        num_input_tokens = response.usage.prompt_tokens
        num_output_tokens = response.usage.completion_tokens
        return self.clean_text(gpttext), num_input_tokens, num_output_tokens

    def clean_text(self, text):
        text = text.replace("$", "").replace("```", "").replace("<<", "").replace(">>", "").replace("**", "")
        text = re.sub(r"\b\w*__\w*\b", "", text)
        text = re.sub("\n+", "\n", text)
        return text


class QAVerbalizer:
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

        self.prof_system_message = (
            "You are a college professor, known for your expert knowledge in deep learning field. "
            "You are also known for creating very thoughtful and probing questions that examine "
            "the actual knowledge of a student based on their submitted paper. Your goal is to come up with "
            "a list of questions, both on intuitive level and on deeper technical level that evaluate if "
            "a student really knows about his or her work. Focus on the knowledge of the main proposed method, "
            "motivation and results. Make sure your list of questions examine the student thoroughly. "
            "Ask at least 10 different and diverse questions. "
            "The questions must cover intuition, main idea and technical details, among others. "
            "Be extremely specific and ask about details presented in the paper, no generic or abstract questions. "
        )

        self.student_system_message = (
            "You are a student, who wrote this paper. You are on a very important exam. "
            "You are tasked to explain your work as best as you can. "
            "You will be provided with a text of the paper, split by pages and a question. "
            "You must answer the question using information given in the paper. "
            "The answer should be consice and to the point but still contain details. "
            "And it should answer the question as best as possible. Be extremly specific. "
            "Ground your response to the provided paper text. Do NOT use generic or abstract phrases. "
            "Your career depends on how well you do this job. I will tip you $2000 for an excellent job done. "
            "Make sure to answer using at least 10 (ten) sentences."
        )

        self.question_schema = {
            "name": "ask_questions",
            "description": "ask questions about provided arxiv paper",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "description": "the list of questions to be asked",
                        "items": {
                            "type": "string",
                            "description": "individual question, thoughtful and revealing",
                        },
                    },
                },
                "required": ["questions"],
            },
        }

    def prepare_messages(self, paper_text):
        # path = os.path.join(files_dir, "original_text_split_pages.txt")
        # with open(path) as f:
        #    paper_text = f.read()

        human_message = f"Below is the student arxiv paper about which the questions needs to be asked: {paper_text}"

        messages = [{"role": "system", "content": self.prof_system_message}, {"role": "user", "content": human_message}]

        response = self.llm_api(
            model=self.llm_strong,
            messages=messages,
            temperature=0,
            functions=[self.question_schema],
            function_call="auto",
        )

        Qs = json.loads(response.choices[0].message.function_call.arguments)

        for Q in Qs["questions"]:
            human_message = (
                f"Here is the text of the split by pages: {paper_text}. And here is the question you need to answer: {Q}. "
                "Make sure your answer best reflects the provided text."
            )

            messages = [
                {"role": "system", "content": self.student_system_message},
                {"role": "user", "content": human_message},
            ]
            yield Q, messages

    def gpts(self, questions):
        for messages in questions:
            response = self.llm_api(model=self.llm_base, messages=messages, temperature=0)
            yield response.choices[0].message.content

    def process_answers(self, gpt_answers, paper_text) -> tuple[Any, list]:
        answers = []
        pages = []

        for answer in gpt_answers:
            # response = self.llm_api(model=self.llm_base, messages=messages, temperature=0)
            # answer = response.choices[0].message.content

            answer = answer.replace("$", "").replace("```", "").replace("<<", "").replace(">>", "").replace("**", "")

            # remove words with underscores
            answer = re.sub(r"\b\w*__\w*\b", "", answer)

            answers.append(answer)

            T = paper_text.split("PAGE ")
            G = answer.split(".")
            seq = self.matcher.match(G[:-1], T[1:], minilm=1, bert=1, fuzz=1, spacy=1, diff=1, tfidf=1, pnt=True)
            # counts = np.bincount(seq)
            # pages.append(np.argmax(counts))
            pages.append(seq)

        return answers, pages


class DocumentProcessor:
    def __init__(self, args):
        self.args = args
        self.files_dir = None
        self.tts_client = None
        self.logging = None
        self.gdrive_client = None
        self.misc_files = []

        self.setup_logging(args)

        if self.args.tts_client == "openai":
            self.tts_client = OpenAITTSClient()
        else:
            self.tts_client = GoogleTTSClient()

        if self.args.gdrive_id:
            self.gdrive_client = GDrive(self.args.gdrive_id)

        openai.api_key = self.args.openai_key
        self.llm_api = openai.chat.completions.create

    def pickle_dump(self, data, filename):
        with open(filename, "wb") as f:
            pickle.dump(data, f)
            self.misc_files.append(filename)

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

        return main_file, self.files_dir

    def process_html(self, main_file) -> tuple[str, str, str]:
        assert self.files_dir is not None
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

        with open(os.path.join(self.files_dir, "extracted_sections.json"), "w") as f:
            json.dump(D, f, indent=4)

        print(json.dumps(D, indent=4))

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

        self.pickle_dump(pagemap, os.path.join(self.files_dir, "map_pagemap.pkl"))
        self.pickle_dump(pageblockmap, os.path.join(self.files_dir, "map_pageblockmap.pkl"))

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
        chunk_audio_file_name = f"question_{hash(text)}.mp3" if is_question else f"answer_{hash(text)}.mp3"

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
                mp3_list_file.write(f"file {file_name}\n")

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
def mock_llm_api(*args, **kwargs):
    from unittest.mock import Mock

    replace = [
        "The text must be written in the first person plural point of view. Do not use long latex",
        "expressions, paraphrase them or summarize in words. Be as faithful to the given text as",
        "possible. Below is the section of the paper, requiring paraphrasing and simplification",
        "and it is indicated by double angle brackets <<",
        ">>. Start with 'In this section, we' and continue in first person plural point of view.",
    ]

    # Create the mock response
    mock_response = Mock()

    # Mock the choices[0].message.content
    choice_mock = Mock()
    messages = kwargs.get("messages", [])
    combined_message = messages[-1]["content"]

    for r in replace:
        combined_message = combined_message.replace(r, "")

    choice_mock.content = combined_message
    mock_response.choices = [Mock(message=choice_mock)]

    # Mock the usage.prompt_tokens and usage.completion_tokens
    usage_mock = Mock()
    usage_mock.prompt_tokens = len(combined_message.split())
    usage_mock.completion_tokens = len(combined_message.split())
    mock_response.usage = usage_mock

    return mock_response


def test():
    # read text and pageblockmap from file
    with open(".temp/1706.03762_files/extracted_orig_text_clean.txt", "r") as f:
        text = f.read()
    with open(".temp/1706.03762_files/map_pageblockmap.pkl", "rb") as f:
        pageblockmap = pickle.load(f)

    # with open(".temp/test_process_sections.pkl", "rb") as f:
    #   results = pickle.load(f)

    import logging
    from map.utils import Matcher
    from pprint import pprint
    from itertools import groupby
    from operator import itemgetter

    verbalizer = Verbalizer(args, Matcher(args.cache_dir), logging)
    verbalizer.llm_strong = "gpt-3.5-turbo-0125"
    verbalizer.llm_base = "gpt-3.5-turbo-0125"
    verbalizer.llm_api = mock_llm_api
    message_batches = list(verbalizer.generate_messages(text, pageblockmap))
    # pprint(message_batches)

    gpt_responses = verbalizer.get_gpt_responses(message_batches)

    gpttext, gptpagemap, verbalizer_steps, textpagemap = verbalizer.process_results(gpt_responses)

    # pprint(gptpagemap)
    pprint(gptpagemap)
    print(len(gptpagemap))

    print(len(sent_tokenize(gpttext)))

    # results = [verbalizer.process_section(sec, pagemap_sections[i]) for i, sec in enumerate(sections)]

    # save results
    # with open(".temp/test_process_sections.pkl", "wb") as f:
    #   pickle.dump(results, f)

    # gpttext, gptpagemap, verbalizer_steps, textpagemap = verbalizer.process_results(results)

    # pprint(gptpagemap)
    # pprint(textpagemap)


def test_qa():
    with open(".temp/1706.03762_files/original_text_split_pages.txt") as f:
        paper_text = f.read()

    p = DocumentProcessor(args)
    p.files_dir = ".temp/debug"
    qa_verbalizer = QAVerbalizer(args, Matcher(args.cache_dir), logging)
    messages = list(qa_verbalizer.prepare_messages(paper_text))
    questions = [q for q, _ in messages]
    gpts = [m for _, m in messages]
    answers = list(qa_verbalizer.gpts(gpts[:1]))
    answers, pages = qa_verbalizer.process_answers(answers[:1], paper_text)

    pprint(questions[:1])
    pprint(answers)

    p.process_qa_speech(questions[:1], answers[:1], "title")


# test_qa()

# test()
