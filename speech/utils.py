from nltk.tokenize import sent_tokenize
from google.cloud import texttospeech
import os
import textwrap
import shutil

from openai import OpenAI, audio


# Abstract class to define TTS client behavior
class TTSClient:
    def synthesize_speech(self, text: str, voice: str, rate: float = 1.0, files_dir: str = None, file_name: str = None):
        """Synthesize speech from text using a specific voice."""
        raise NotImplementedError("Must implement synthesize_speech method.")


class GoogleTTSClient(TTSClient):
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()

    def synthesize_speech(self, text: str, voice: str, rate: float = 1.0, files_dir=None, file_name=None):
        voice_params = texttospeech.VoiceSelectionParams(language_code="en-US", name=f"en-US-{voice}")
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=rate)

        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            response = self.client.synthesize_speech(
                input=synthesis_input, voice=voice_params, audio_config=audio_config
            )
            audio_content = response.audio_content

            if files_dir and file_name:
                file_path = os.path.join(files_dir, file_name)
                with open(file_path, "wb") as out:
                    out.write(audio_content)
            else:
                file_path = None

            return audio_content, file_path
        except Exception as e:
            return None, None


class OpenAITTSClient(TTSClient):
    def __init__(self):
        self.client = OpenAI()

    def synthesize_speech(
        self, text, voice, rate=1.0, files_dir=None, file_name=None
    ) -> tuple[bytes, str] | tuple[None, None]:

        voices = {"Polyglot-1": "onxy", "Studio-O": "alloy", "Studio-Q": "shimmer", "onyx": "onyx"}

        response = self.client.audio.speech.create(
            model="tts-1", voice=voices.get(voice, "onyx"), input=text, speed=rate
        )

        audio_content = b""
        if files_dir and file_name:
            file_path = os.path.join(files_dir, file_name)
            with open(file_path, mode="wb") as f:
                for data in response.iter_bytes():
                    f.write(data)
                    audio_content += data
            return audio_content, file_path
        else:
            return None, None


def text_to_speech_qa(questions, answers, mp3_list_file, files_dir, tts_client: TTSClient, ffmpeg, logging):

    for q, a in zip(questions, answers):
        chunk_audio_file_name = f"question_{hash(q)}.mp3"
        audio_content, chunk_audio = tts_client.synthesize_speech(q, "Studio-O", 1.0, files_dir, chunk_audio_file_name)

        logging.info(f"Processed question: \n\n {q}")

        mp3_list_file.write(f'file {chunk_audio_file_name}\n')

        # append 2 seconds
        chunk_audio_with_silence = os.path.join(files_dir, f'q_with_silence_{hash(q)}.mp3')
        os.system(f'{ffmpeg} -i {chunk_audio} -f lavfi -t 2 -i anullsrc=r=44100:cl=stereo -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1" {chunk_audio_with_silence}')
        shutil.move(chunk_audio_with_silence, chunk_audio)

        a_sent = a.split('.')[:-1]

        for a_s in a_sent:
            chunk_audio_file_name = f"answer_{hash(a_s)}.mp3"
            audio_content, file_path = tts_client.synthesize_speech(
                a_s, "Studio-Q", 1.0, files_dir, chunk_audio_file_name
            )

            mp3_list_file.write(f'file {chunk_audio_file_name}\n')

        logging.info(f'Processed answer: \n\n {a}')
        logging.info("-" * 100)


def text_to_speech_short(text, slides, mp3_list_file, files_dir, tts_client: TTSClient, logging):
    para = text.split('\n\n')
    slides = slides['slides']

    if len(para) == len(slides):
        first_para_sent = sent_tokenize(para[0])
        para[0] = " ".join(first_para_sent[1:])
        para.insert(0, first_para_sent[0])

    for s in para:
        chunk_audio_file_name = f"short_{hash(s)}.mp3"
        audio, chunk_audio = tts_client.synthesize_speech(s, "Neural2-F", 1.0, files_dir, chunk_audio_file_name)

        logging.info(f'Processed block text: \n\n {s}')
        logging.info("-" * 100)

        if os.path.getsize(chunk_audio) == 0:
            continue

        mp3_list_file.write(f'file {chunk_audio_file_name}\n')


def text_to_speechvideo(text, mp3_list_file, files_dir, tts_client: TTSClient, pageblockmap, voice, logging):
    splits = sent_tokenize(text)

    assert len(pageblockmap) == len(splits), "Number of pageblockmap does not match number of splits"

    block_text = []
    prev = pageblockmap[0]
    last_page = 0

    for ind, m in enumerate(pageblockmap):
        if m == prev:
            block_text.append(splits[ind])
            continue

        joinedtext = ' '.join(block_text)
        if isinstance(prev, list):
            last_page = prev[1]
            synthesize(
                joinedtext,
                tts_client,
                files_dir,
                mp3_list_file,
                page=prev[1],
                block=prev[2],
                voice=voice,
                logging=logging,
            )
        else:
            synthesize(joinedtext, tts_client, files_dir, mp3_list_file, page=last_page, voice=voice, logging=logging)

        prev = m
        block_text = [splits[ind]]


def synthesize(text, tts_client: TTSClient, files_dir, mp3_list_file, page=None, block=None, voice=None, logging=None):
    if block is None:
        chunk_audio_file_name = f"page{page}summary_{hash(text)}.mp3"
    else:
        chunk_audio_file_name = f"page{page}block{block}_{hash(text)}.mp3"

    audio, _ = tts_client.synthesize_speech(text, voice, 1.0, files_dir, chunk_audio_file_name)

    # if processing fails, subdivide into smaller chunks and try again
    if audio:
        logging.info(f'Processed block text: \n\n {text}')
        logging.info("-" * 100)

        mp3_list_file.write(f'file {chunk_audio_file_name}\n')
    else:
        chunks = textwrap.wrap(text,
                               width=len(text) // 2,
                               break_long_words=False,
                               expand_tabs=False,
                               replace_whitespace=False,
                               drop_whitespace=False,
                               break_on_hyphens=False)

        synthesize(chunks[0], tts_client, files_dir, mp3_list_file, page, block, voice, logging)
        synthesize(chunks[1], tts_client, files_dir, mp3_list_file, page, block, voice, logging)


def text_to_speech(text, mp3_list_file, files_dir, tts_client: TTSClient, voice, logging):
    splits = sent_tokenize(text)

    block = []

    for s in splits:

        block.append(s)
        if len(block) < 3:
            continue

        block_text = ' '.join(block)
        chunk_audio_file_name = f"part_{hash(block_text)}.mp3"
        _, _ = tts_client.synthesize_speech(block_text, voice, 1.0, files_dir, chunk_audio_file_name)

        logging.info(f'Processed text: \n\n {block_text}')
        logging.info("-" * 100)

        mp3_list_file.write(f"file {chunk_audio_file_name}\n")
        block = []
