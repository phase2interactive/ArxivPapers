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


def setup_logging(args):
    if args.verbose == "debug":
        level = logging.DEBUG
    elif args.verbose == "info":
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(level=level, format="\n%(asctime)s - %(levelname)s - %(message)s")


def process_latex(args):
    paper_id = args.paperid
    remove_oldfiles_samepaper(paper_id)
    files_dir = download_paper(paper_id)
    tex_files = get_tex_files(files_dir)
    logging.info(f"Found .tex files: {tex_files}")
    main_file = get_main_source_file(tex_files, files_dir)

    if main_file:
        logging.info(f"Found [{main_file}.tex] as a main source file")
    else:
        logging.info("Failed to find main source file. Enter manually:")
        main_file = input()

    return files_dir, main_file


def process_html(args, files_dir, main_file):
    html_parser = "latex2html" if args.l2h else "latexmlc"
    display = "> /dev/null 2>&1" if args.verbose != "debug" else ""

    generate_html(
        main_file, files_dir, html_parser, args.pdflatex, args.latex2html, args.latexmlc, args.gs, display, logging
    )

    citations = parse_aux_file(os.path.join(files_dir, f"{main_file}.aux"))
    title, content, html, abstract = parse_html_file(
        os.path.join(files_dir, main_file, f"{main_file}.html"), html_parser
    )

    logging.info(f"Parsed HTML file. Title: {title}")

    stop_words = ["references", "appendix", "conclusion", "acknowledgments", "about this document"]
    if args.stop_word:
        stop_words.append(args.stop_word)

    logging.info(f"Section title stop words: {stop_words}")

    D = create_nested_dict(content, logging, stop_words)
    extract_text_recursive(D, files_dir, main_file, citations, html_parser, html)
    text = depth_first_search(D)

    splits = sent_tokenize(text)
    text = " ".join(splits)

    with open(os.path.join(files_dir, "extracted_orig_text_clean.txt"), "w") as f:
        f.write(text)

    with open(os.path.join(files_dir, "original_text_split_sections.txt"), "w") as f:
        sections = text.split(" Section: ")
        for i, s in enumerate(sections):
            f.write(f"SECTION {i + 1}\n\n")
            f.write(s)
            f.write("\n\n")

    return title, text, abstract


def process_map(args, files_dir, main_file, text):
    if not args.create_video:
        return None, None

    matcher = Matcher(args.cache_dir)

    logging.info("Mapping text to pages")
    pagemap = map_text_to_pdfpages(text, f"{os.path.join(files_dir, main_file)}.pdf", matcher)

    logging.info("Mapping pages to blocks")
    coords, pageblockmap = map_page_to_blocks(
        pagemap, text, args.gs, files_dir, f"{os.path.join(files_dir, main_file)}.pdf", matcher, "> /dev/null 2>&1"
    )

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


def process_gpt(args, files_dir, text, pagemap, pageblockmap, title, llm_api, matcher):
    tmpdata = {}

    if args.create_short:
        gpttext_short, slides_short = gpt_short_verbalizer(files_dir, llm_api, args.llm_strong, args.llm_base, logging)
        with open(os.path.join(files_dir, "gpt_text_short.txt"), "w") as f:
            f.write(gpttext_short)

        with open("gpt_slides_short.json", "w") as json_file:
            json.dump(slides_short, json_file, indent=4)

        tmpdata["gpttext_short"] = gpttext_short
        tmpdata["gptslides_short"] = slides_short["slides"]

    if args.create_qa:
        questions, answers, qa_pages = gpt_qa_verbalizer(files_dir, llm_api, args.llm_base, matcher, logging)

        create_questions(questions, os.path.join(files_dir, "questions"))

        with open(os.path.join(files_dir, "qa_pages.pkl"), "wb") as f:
            pickle.dump(qa_pages, f)

        with open(os.path.join(files_dir, "gpt_questions_answers.txt"), "w") as f:
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

    if args.create_video:
        gpttext, gptpagemap, verbalizer_steps, textpagemap = gpt_textvideo_verbalizer(
            text,
            llm_api,
            args.llm_strong,
            args.llm_base,
            args.manual_gpt,
            args.include_summary,
            pageblockmap,
            matcher,
            logging,
        )

        with open(os.path.join(files_dir, "gptpagemap.pkl"), "wb") as f:
            pickle.dump(gptpagemap, f)

        tmpdata.update(
            {
                "gpttext": gpttext,
                "gptpagemap": gptpagemap,
                "verbalizer_steps": verbalizer_steps,
                "textpagemap": textpagemap,
            }
        )

        with open(os.path.join(files_dir, "gpt_verb_steps.txt"), "w") as f:
            for si, s in enumerate(verbalizer_steps):
                f.write(f"===Original {si}===\n\n")
                f.write(s[0])
                f.write("\n\n")
                f.write(f"===GPT {si}===\n\n")
                f.write(s[1])
                f.write("\n\n")

        with open(os.path.join(files_dir, "gpt_text.txt"), "w") as f:
            f.write(gpttext)

        logging.info(f"Extracted text:\n\n {gpttext}")

    if args.create_audio_simple:
        gpttext, verbalizer_steps = gpt_text_verbalizer(
            text, llm_api, args.llm_base, args.manual_gpt, args.include_summary, logging
        )

    if len(tmpdata) > 0:
        with open(os.path.join(files_dir, "tmpdata.pkl"), "wb") as f:
            pickle.dump(tmpdata, f)

    return gpttext, gptpagemap if args.create_video else None


def process_speech(args, files_dir, gpttext, gpttext_short, slides_short, questions, answers, gptpagemap, title):
    if args.tts_client == "openai":
        tts_client = OpenAITTSClient()
    else:
        tts_client = GoogleTTSClient()

    if args.gdrive_id:
        gdrive_client = GDrive(args.gdrive_id)

    if args.create_short:
        process_short_speech(
            gpttext_short,
            slides_short,
            files_dir,
            tts_client,
            logging,
            args,
            gdrive_client,
            title,
        )

    if args.create_qa:
        process_qa_speech(
            questions,
            answers,
            files_dir,
            tts_client,
            args,
            logging,
            gdrive_client,
            title,
        )

    if args.create_video:
        process_video_speech(
            gpttext,
            files_dir,
            tts_client,
            gptpagemap,
            args,
            logging,
        )

    if args.create_audio_simple:
        process_simple_speech(
            gpttext,
            files_dir,
            tts_client,
            args,
            logging,
        )


def process_short_speech(gpttext_short, slides_short, files_dir, tts_client, logging, args, gdrive_client, title):
    with open(os.path.join(files_dir, args.chunk_mp3_file_list), "w") as mp3_list_file:
        text_to_speech_short(gpttext_short, slides_short, mp3_list_file, files_dir, tts_client, logging)

    shutil.copy(
        os.path.join(files_dir, args.chunk_mp3_file_list),
        os.path.join(files_dir, f"shorts_{args.chunk_mp3_file_list}"),
    )

    final_audio_short = os.path.join(files_dir, f"{args.final_audio_file}_short.mp3")
    os.system(
        f"{args.ffmpeg} -f concat -i {os.path.join(files_dir, args.chunk_mp3_file_list)} "
        f"-c copy {final_audio_short} > /dev/null 2>&1"
    )

    logging.info("Created short audio file")

    if args.gdrive_id:
        gdrive_client.upload_audio(f"[short] {title}", f"{final_audio_short}")
        logging.info("Uploaded short audio to GDrive")

    create_slides(slides_short, os.path.join(files_dir, "slides"))


def process_qa_speech(questions, answers, files_dir, tts_client, args, logging, gdrive_client, title):
    with open(os.path.join(files_dir, args.chunk_mp3_file_list), "w") as mp3_list_file:
        text_to_speech_qa(questions, answers, mp3_list_file, files_dir, tts_client, args.ffmpeg, logging)

    shutil.copy(
        os.path.join(files_dir, args.chunk_mp3_file_list),
        os.path.join(files_dir, f"qa_{args.chunk_mp3_file_list}"),
    )

    final_audio_qa = os.path.join(files_dir, f"{args.final_audio_file}_qa.mp3")
    os.system(
        f"{args.ffmpeg} -f concat -i {os.path.join(files_dir, args.chunk_mp3_file_list)} "
        f"-c copy {final_audio_qa} > /dev/null 2>&1"
    )

    logging.info("Created QA audio file")

    if args.gdrive_id:
        gdrive_client.upload_audio(f"[QA] {title}", f"{final_audio_qa}")
        logging.info("Uploaded QA audio to GDrive")


def process_video_speech(gpttext, files_dir, tts_client, gptpagemap, args, logging):
    with open(os.path.join(files_dir, args.chunk_mp3_file_list), "w") as mp3_list_file:
        text_to_speechvideo(gpttext, mp3_list_file, files_dir, tts_client, gptpagemap, args.voice, logging)


def process_simple_speech(gpttext, files_dir, tts_client, args, logging):
    with open(os.path.join(files_dir, args.chunk_mp3_file_list), "w") as mp3_list_file:
        text_to_speech(gpttext, mp3_list_file, files_dir, tts_client, args.voice, logging)


def process_zip(args, files_dir, main_file):
    renamed_main = "main"
    if main_file != renamed_main:
        temp_filename = f"temp_{random.randint(1000, 9999)}"
        shutil.copy(f"{os.path.join(files_dir, main_file)}.pdf", f"{os.path.join(files_dir, temp_filename)}.pdf")
        shutil.copy(f"{os.path.join(files_dir, temp_filename)}.pdf", f"{os.path.join(files_dir, renamed_main)}.pdf")

    crop_pdf(
        f"{files_dir}/{main_file}.pdf",
        f"{files_dir}/fpage.pdf",
        args.gs,
        upper_top=3,
        top_percent=25,
        left_percent=12,
        right_percent=7,
    )

    return zip_files(
        files_dir,
        args.gs,
        args.ffmpeg,
        args.create_short,
        args.create_qa,
        args.create_video,
        args.final_audio_file,
        args.chunk_mp3_file_list,
        "> /dev/null 2>&1",
    )


def main(args):
    setup_logging(args)

    files_dir, main_file = process_latex(args)
    title, text, abstract = process_html(args, files_dir, main_file)
    # if args.extract_text_only:
    #    return None

    pagemap, pageblockmap = process_map(args, files_dir, main_file, text)

    openai.api_key = args.openai_key
    llm_api = openai.chat.completions.create
    matcher = Matcher(args.cache_dir)

    gpttext, gptpagemap = process_gpt(args, files_dir, text, pagemap, pageblockmap, title, llm_api, matcher)

    if args.create_short:
        gpttext_short, slides_short = gpt_short_verbalizer(files_dir, llm_api, args.llm_strong, args.llm_base, logging)
    else:
        gpttext_short, slides_short = None, None

    if args.create_qa:
        questions, answers, _ = gpt_qa_verbalizer(files_dir, llm_api, args.llm_base, matcher, logging)
    else:
        questions, answers = None, None

    process_speech(args, files_dir, gpttext, gpttext_short, slides_short, questions, answers, gptpagemap, title)
    create_summary(abstract, title, args.paperid, llm_api, args.llm_base, files_dir)
    final_audio = os.path.join(files_dir, f"{args.final_audio_file}.mp3")
    os.system(
        f"{args.ffmpeg} -f concat -i {os.path.join(files_dir, args.chunk_mp3_file_list)} -c copy {final_audio} > /dev/null 2>&1"
    )
    logging.info("Created audio file")

    if args.gdrive_id:
        gdrive_client = GDrive(args.gdrive_id)
        gdrive_client.upload_audio(title, f"{final_audio}")
        logging.info("Uploaded audio to GDrive")

    return process_zip(args, files_dir, main_file)


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
    voice="Polyglot-1",
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

main(args)
