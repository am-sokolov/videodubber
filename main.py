import argparse
import os
import uuid
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate
import whisper
import spacy
from spacy_syllables import SpacySyllables
from tqdm import tqdm

# You can add more Spacy models from https://spacy.io/models
spacy_models = {
    "en": "en_core_web_sm",
    "english": "en_core_web_sm",
    "de": "de_core_news_sm",
    "german": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "french": "fr_core_news_sm",
    "it": "it_core_news_sm",
    "italy": "it_core_news_sm"
}


def extract_audio_from_video(video_file):
    try:
        print("Extracting audio track")
        video = VideoFileClip(video_file)
        audio = video.audio
        audio_file = os.path.splitext(video_file)[0] + ".wav"
        audio.write_audiofile(audio_file)
        return audio_file
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return None


def transcribe_audio(audio_file, source_language):
    try:
        print("Transcribing audio track")
        model = whisper.load_model("large")
        trans = model.transcribe(audio_file, language=source_language, verbose=False, word_timestamps=True)
        return trans
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None


def translate_text(texts, target_language):
    try:
        translate_client = translate.Client()
        results = translate_client.translate(texts, target_language=target_language)
        return [result['translatedText'] for result in results]
    except Exception as e:
        print(f"Error translating texts: {e}")
        return None


def create_audio_from_text(text, target_language, target_voice):
    audio_file = "translated_" + str(uuid.uuid4()) + ".wav"
    try:
        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=target_language,
            name=target_voice
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )
        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )
        with open(audio_file, "wb") as out:
            out.write(response.audio_content)
        return audio_file
    except Exception as e:
        if os.path.isfile(audio_file):
            os.remove(audio_file)
        raise Exception(f"Error creating audio from text: {e}")


ABBREVIATIONS = {
    "Mr.": "Mister",
    "Mrs.": "Misses",
    "No.": "Number",
    "Dr.": "Doctor",
    "Ms.": "Miss",
    "Ave.": "Avenue",
    "Blvd.": "Boulevard",
    "Ln.": "Lane",
    "Rd.": "Road",
    "a.m.": "before noon",
    "p.m.": "after noon",
    "ft.": "feet",
    "hr.": "hour",
    "min.": "minute",
    "sq.": "square",
    "St.": "street",
    "Asst.": "assistant",
    "Corp.": "corporation"
}

STOP_WORDS = [
    "APPLAUSE",
    "APLAUSOS",
    "BEIFALL",
    "APPLAUDISSEMENTS",
    "APPLAUSI"
]


def merge_audio_files(transcription, source_language, target_language, target_voice):
    temp_files = []
    try:
        if spacy_models[source_language] not in spacy.util.get_installed_models():
            spacy.cli.download(spacy_models[source_language])
        nlp = spacy.load(spacy_models[source_language])
        nlp.add_pipe("syllables", after="tagger")
        merged_audio = AudioSegment.silent(duration=0)
        sentences = []
        sentence_starts = []
        sentence_ends = []
        sentence = ""
        sent_start = 0
        print("Composing sentences")
        for segment in tqdm(transcription["segments"]):
            for i, word in enumerate(segment["words"]):
                word["word"] = ABBREVIATIONS.get(word["word"].strip(), word["word"])
                if any(stop_word in word["word"] for stop_word in
                       STOP_WORDS):  # trick to round about the whisper model flaw (only for English)
                    continue
                if sent_start == 0:
                    sent_start = word["start"]
                sentence += word["word"] + " "
                # this is a hack to compensate the absense of VAD in Whisper
                if i == len(segment["words"]) - 1:  # last word in segment
                    word_speed = sum(
                        token._.syllables_count for token in nlp(word["word"]) if token._.syllables_count) / (
                                         word["end"] - word["start"])
                    segment_speed = sum(
                        token._.syllables_count for token in nlp(segment["text"]) if token._.syllables_count) / (
                                            segment["end"] - segment["start"])
                    if word_speed < 1.0 or segment_speed < 1.0:
                        word["word"] += "."
                if word["word"].endswith("."):
                    sentences.append(sentence)
                    sentence_starts.append(sent_start)
                    sentence_ends.append(word["end"])
                    sent_start = 0
                    sentence = ""
        # translate sentences in chunks of 128
        print("Translating sentences")
        translated_texts = []
        for i in tqdm(range(0, len(sentences), 128)):
            chunk = sentences[i:i + 128]
            translated_chunk = translate_text(chunk, target_language)
            if translated_chunk is None:
                raise Exception("Translation failed")
            translated_texts.extend(translated_chunk)
        print("Creating translated audio track")
        for i, translated_text in enumerate(tqdm(translated_texts)):
            translated_audio_file = create_audio_from_text(translated_text, target_language, target_voice)
            if translated_audio_file is None:
                raise Exception("Audio creation failed")
            temp_files.append(translated_audio_file)
            translated_audio = AudioSegment.from_wav(translated_audio_file)
            original_duration = int(sentence_ends[i] * 1000)
            new_duration = len(translated_audio) + len(merged_audio)
            padding_duration = max(0, original_duration - new_duration)
            padding = AudioSegment.silent(duration=padding_duration)
            merged_audio += padding + translated_audio
        return merged_audio
    except Exception as e:
        print(f"Error merging audio files: {e}")
        return None
    finally:
        # cleanup: remove all temporary files
        for file in temp_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error removing temporary file {file}: {e}")


def save_audio_to_file(audio, filename):
    try:
        audio.export(filename, format="wav")
    except Exception as e:
        print(f"Error saving audio to file: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to the source video file')
    parser.add_argument('target_voice', type=str,
                        help='Target dubbing voice name from https://cloud.google.com/text-to-speech/docs/voices')
    parser.add_argument('credentials', type=str, help='Path to the Google Cloud credentials JSON file')
    parser.add_argument('source_language', type=str, help='Source language, e.g. english')
    args = parser.parse_args()

    # Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.credentials

    audio_file = extract_audio_from_video(args.input)
    if audio_file is None:
        return

    transcription = transcribe_audio(audio_file, args.source_language)
    if transcription is None:
        return

    merged_audio = merge_audio_files(transcription, args.source_language, args.target_voice[:5], args.target_voice)
    if merged_audio is None:
        return

    # Save the audio file with the same name as the video file but with a ".wav" extension
    output_filename = os.path.splitext(args.input)[0] + ".wav"
    save_audio_to_file(merged_audio, output_filename)


if __name__ == "__main__":
    main()
