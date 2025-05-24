# import soundfile as sf
import sounddevice as sd
# import vosk
import pyaudio
import json
import logging
from RealtimeSTT import AudioToTextRecorder

from misaki import en, espeak

from kokoro_onnx import Kokoro

import anthropic
from anthropic import Anthropic

AUDIO_SAMPLE_RATE = 16000

AGENT_NAME = "John"
KOTORO_VOICE = "am_michael"

logging.basicConfig(level=logging.CRITICAL)


class AnthropicModelInterface:
    def __init__(self, model_prompt_path="model-prompt.txt"):
        with open(model_prompt_path, "r") as file:
            self.system_prompt = file.read()

        with open("anthropic-api-key.txt", "r") as keyfile:
            anthropic_api_key = keyfile.read().rstrip()
        if (not anthropic_api_key):
            print("No key provided")
            exit()

        self.client = Anthropic(api_key=anthropic_api_key)
        self.conversation_log = []

    def MakeAnthropicAPICall(self):
        try:
            message = self.client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=1024,
                temperature=1,
                system=self.system_prompt,
                messages=self.conversation_log
            )
            return message.content[0].text
        except anthropic.APIConnectionError as e:
            print("The server could not be reached")
            # an underlying Exception, likely raised within httpx.
            print(e.__cause__)
            return ""
        except anthropic.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
            return ""
        except anthropic.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
            return ""

    def AddUserMessage(self, text):
        self.conversation_log.append({
            "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
        })

    def AddAgentMessage(self, text):
        self.conversation_log.append({
            "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
        })


def WriteJsonFile(string):
    try:
        f = json.loads(string)
        if (not f["file"] or not f["text"]):
            print("Wrong fields in JSON")
            return
        # Write mode - Overwrites the file if it exists
        with open(f["file"], "w") as file:
            file.write(f["text"])
    except json.JSONDecodeError as e:
        print(f"Malformed JSON: {string}")
        print(e)


def HandleFileCreationString(string):
    in_json = False
    idx = 0
    start_idx = 0
    json_to_replace = []

    for idx, value in enumerate(string):
        if (not in_json and value == "{"):
            in_json = True
            start_idx = idx
            continue
        elif (in_json and value == "}"):
            in_json = False
            json_content = string[start_idx:(idx+1)]
            WriteJsonFile(json_content)
            json_to_replace.append(json_content)

    scrubbed_string = string
    for json_string in json_to_replace:
        scrubbed_string = scrubbed_string.replace(json_string, "")

    return scrubbed_string


def CheckForTerminate(string):
    return "It was good talking with you." in string


def PrintAgent(string):
    print(f"{AGENT_NAME} says: \n \"{string}\" \n")


def PrintUser(string):
    print(f"User says: \n \"{string}\" \n")


if __name__ == "__main__":

    print("Setting Up")
    # Misaki G2P with espeak-ng fallback
    fallback = espeak.EspeakFallback(british=False)
    g2p = en.G2P(trf=False, british=False, fallback=fallback)

    # Kokoro Text to Speech
    kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")

    # # Vosk Speech to Text
    # vosk.SetLogLevel(-1)
    # model = vosk.Model("models/vosk-model-en-us-0.22")
    # rec = vosk.KaldiRecognizer(model, AUDIO_SAMPLE_RATE)

    recorder = AudioToTextRecorder()
    anthropic_interface = AnthropicModelInterface()

    print("\033[H\033[J", end="")
    print(f"{AGENT_NAME} has joined")

    messages = []

    # AddUserMessage("hello!", messages)
    # response = MakeAnthropicAPICall(client, messages)
    # PrintAgent(response)
    # AddAgentMessage(response, messages)

    # Phonemize
    phonemes, _ = g2p("Hello!")
    # Create
    samples, sample_rate = kokoro.create(
        phonemes, KOTORO_VOICE, is_phonemes=True)
    # Play
    sd.play(samples, sample_rate)
    sd.wait()

    # Open the microphone stream
    p = pyaudio.PyAudio()
    # print(p.get_device_info_by_index(1))
    # exit()
    try:
        while True:
            heard_text = ""
            print(f"{AGENT_NAME} is listening. Press any key to stop listening")
            
            recorder.start()
            input("")
            recorder.stop()
            heard_text = recorder.text()

            if (not heard_text):
                # Phonemize
                phonemes, _ = g2p("It seems you didn't say anything.")
                # Create
                samples, sample_rate = kokoro.create(
                    phonemes, KOTORO_VOICE, is_phonemes=True)
                # Play
                sd.play(samples, sample_rate)
                sd.wait()
                continue

            PrintUser(heard_text)
            anthropic_interface.AddUserMessage(heard_text)
            response = anthropic_interface.MakeAnthropicAPICall()
            anthropic_interface.AddAgentMessage(response)

            text_to_say = (HandleFileCreationString(response))
            PrintAgent(text_to_say)
            # Phonemize
            if (text_to_say):
                phonemes, _ = g2p(text_to_say)

                # Create
                samples, sample_rate = kokoro.create(
                    phonemes, KOTORO_VOICE, is_phonemes=True)

                # Play
                sd.play(samples, sample_rate)
                sd.wait()

                if CheckForTerminate(text_to_say):
                    exit()

    except KeyboardInterrupt:
        exit()
