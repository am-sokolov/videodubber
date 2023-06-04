
# Video Dubber

This Python script extracts the audio from a video file, transcribes it, translates it into a different language, and then generates a new audio file with the translated text.

## Prerequisites

- Python 3.8 or higher
- [FFmpeg](https://ffmpeg.org/download.html)

## Technologies Used

- [Google Cloud Text-to-Speech API](https://cloud.google.com/text-to-speech): Used to generate the audio for the translated text.
- [Google Cloud Translate API](https://cloud.google.com/translate): Used to translate the transcribed text into a different language.
- [Whisper ASR](https://www.openai.com/research/whisper/): Used to transcribe the audio from the video file.
- [Spacy](https://spacy.io/): Used for natural language processing tasks, such as tokenization and syllable counting.
- [PyDub](http://pydub.com/): Used for manipulating audio files.
- [MoviePy](https://zulko.github.io/moviepy/): Used for extracting the audio from the video file.


## Installation

1. Clone this repository:
   ```
   git clone https://github.com/am-sokolov/videodubber.git
   ```
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Google Cloud Credentials

This script uses Google Cloud's Text-to-Speech and Translate APIs, which require authentication. Follow these steps to get your credentials:

1. Create a new project in the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the [Text-to-Speech](https://cloud.google.com/text-to-speech/docs/quickstart-client-libraries) and [Translate](https://cloud.google.com/translate/docs/setup) APIs for your project.
3. Create a new service account for your project in the [Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts) page.
4. Create a new JSON key for your service account, and download it. This is your credentials file.

## Usage

```
python main.py <input_video_file> <target_voice> <path_to_credentials> <source_language>
```

- `<input_video_file>`: Path to the source video file.
- `<target_voice>`: Target dubbing voice name from [Google Text-to-Speech Voices](https://cloud.google.com/text-to-speech/docs/voices).
- `<path_to_credentials>`: Path to the Google Cloud credentials JSON file.
- `<source_language>`: Source language (e.g., "english").

The script will create a new `.wav` audio file with the same name as the input video file.

## Testing

You can test this script with any video that contains narration. For example, you can use this [free video of US President Donald Trump speaking at the Young Black Leadership Summit at the White House](https://www.videvo.net/video/us-president-donald-trump-speaks-to-african-americans-young-black-leadership-summit-at-the-white-house-8/613121/). 

Here are the step-by-step instructions for testing:

1. Download the video from the link above. 

2. Save the video file in the same directory as the script under the name `trump_speech.mp4`.

3. Run the script with the downloaded video file as the input. For example, if you saved the video as `trump_speech.mp4`, you would run:

   ```
   python main.py trump_speech.mp4  de-DE-Neural2-B path_to_credentials.json  english
   ```

   Replace `path_to_credentials.json` with the path to your Google Cloud credentials JSON file.

4. The script will create a new `.wav` audio file named `trump_speech.wav` in the same directory. This file contains the translated audio.

5. Listen to the `trump_speech.wav` file to verify that the script worked correctly. The audio should be a translation of the original speech in the video.

Please replace `de-DE-Neural2-B` with the desired target voice.

## License

Alexey Sokolov (c). This project is licensed under the terms of the MIT license.
