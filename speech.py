import sounddevice as sd    # Audio playback for generated speech
import pyaudio             # Audio stream handling
import json                # JSON parsing for file creation commands
import logging             # Error logging and debugging
# Real-time speech-to-text conversion
from RealtimeSTT import AudioToTextRecorder

from misaki import en, espeak  # Text-to-phoneme conversion for speech synthesis
from kokoro_onnx import Kokoro  # Neural text-to-speech synthesis

import anthropic           # Anthropic's Claude AI API
from anthropic import Anthropic

# Configuration constants
AUDIO_SAMPLE_RATE = 16000   # Standard sample rate for audio processing (16kHz)
AGENT_NAME = "John"         # Display name for the AI assistant
# Voice model from kotoro. Some examples: am_adam, am_michael, am_eric
KOTORO_VOICE = "am_michael"
# see https://huggingface.co/hexgrad/Kokoro-82M/tree/cddbcb284a842f5679b33f174250190463775a22/voices

# Set logging level to only show critical errors (reduces console noise)
logging.basicConfig(level=logging.CRITICAL)


class AnthropicModelInterface:
    """
    Handles all interactions with the Anthropic Claude AI API
    Manages conversation history, API authentication and and API calls
    """

    def __init__(self, model_prompt_path="model-prompt.txt"):
        """
        Initialize the AI interface with system prompt and API credentials

        Args:
            model_prompt_path (str): Path to file containing the system prompt for Claude
        """
        # Load the system prompt that defines the AI's behavior and personality
        with open(model_prompt_path, "r") as file:
            self.system_prompt = file.read()

        # Load API key from file (keeps credentials separate from code)
        with open("anthropic-api-key.txt", "r") as keyfile:
            anthropic_api_key = keyfile.read().rstrip()

        # Exit if no API key is provided
        if (not anthropic_api_key):
            print("No key provided")
            exit()

        # Initialize the Anthropic client with API key
        self.client = Anthropic(api_key=anthropic_api_key)

        # Store conversation history as a list of message objects
        self.conversation_log = []

    def MakeAnthropicAPICall(self):
        """
        Send the current conversation to Claude and get a response

        Returns:
            str: Claude's response text, or empty string if error occurs
        """
        try:
            # Make API call to Claude with current conversation history
            message = self.client.messages.create(
                model="claude-3-5-haiku-latest",  # Use the Haiku model for faster responses
                max_tokens=1024,                  # Limit response length
                temperature=1,                    # High creativity/randomness
                system=self.system_prompt,        # System instructions for AI behavior
                messages=self.conversation_log    # Full conversation history
            )
            return message.content[0].text

        # Handle various API errors gracefully
        except anthropic.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # Print underlying network error
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
        """
        Add a user message to the conversation history
        """
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
        """
        Add an assistant message to the conversation history
        """
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
    """
    Parse a JSON string and write content to a file
    Expected JSON format: {"file": "filename.txt", "text": "content to write"}

    Args:
        string (str): JSON string containing file path and content
    """
    try:
        # Parse the JSON string
        f = json.loads(string)

        # Validate required fields are present
        if (not f["file"] or not f["text"]):
            print("Wrong fields in JSON")
            return

        # Write the content to the specified file (overwrites if exists)
        with open(f["file"], "w") as file:
            file.write(f["text"])

    except json.JSONDecodeError as e:
        print(f"Malformed JSON: {string}")
        print(e)


def HandleFileCreationString(string):
    """
    Extract JSON file creation commands from AI response and execute them
    Removes the JSON commands from the response text for clean speech output

    Args:
        string (str): AI response that may contain JSON file creation commands

    Returns:
        str: Cleaned response text with JSON commands removed
    """
    in_json = False      # Track if we're inside a JSON block
    idx = 0              # Current character index
    start_idx = 0        # Start of current JSON block
    json_to_replace = []  # List of JSON strings to remove from response

    # Scan through the string character by character
    for idx, value in enumerate(string):
        if (not in_json and value == "{"):
            # Found start of JSON block
            in_json = True
            start_idx = idx
            continue
        elif (in_json and value == "}"):
            # Found end of JSON block
            in_json = False
            json_content = string[start_idx:(idx+1)]

            # Process the JSON command (write file)
            WriteJsonFile(json_content)

            # Mark this JSON for removal from response text
            json_to_replace.append(json_content)

    # Remove all JSON commands from the response text
    scrubbed_string = string
    for json_string in json_to_replace:
        scrubbed_string = scrubbed_string.replace(json_string, "")

    return scrubbed_string


def CheckForTerminate(string):
    """
    Check if the AI response contains a termination phrase

    Args:
        string (str): AI response text to check

    Returns:
        bool: True if termination phrase is found
    """
    return "It was good talking with you." in string


def PrintAgent(string):
    """
    Print the AI agent's message with formatting

    Args:
        string (str): Message to print
    """
    print(f"{AGENT_NAME} says: \n \"{string}\" \n")


def PrintUser(string):
    """
    Print the user's message with formatting

    Args:
        string (str): Message to print
    """
    print(f"User says: \n \"{string}\" \n")


if __name__ == "__main__":
    """
    Main program loop - sets up all components and runs the voice interaction loop
    """

    print("Setting Up")

    # Initialize text-to-phoneme conversion system
    # Misaki G2P (Grapheme-to-Phoneme) with espeak-ng fallback for unknown words
    # Use American English pronunciation
    fallback = espeak.EspeakFallback(british=False)
    g2p = en.G2P(trf=False, british=False, fallback=fallback)

    # Initialize neural text-to-speech system (Kokoro)
    kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")

    # Legacy Vosk speech recognition setup (commented out, replaced by RealtimeSTT)
    # vosk.SetLogLevel(-1)  # Suppress Vosk logging
    # model = vosk.Model("models/vosk-model-en-us-0.22")  # Load speech model
    # rec = vosk.KaldiRecognizer(model, AUDIO_SAMPLE_RATE)  # Create recognizer

    # Initialize modern real-time speech-to-text recorder
    recorder = AudioToTextRecorder()

    # Initialize the Claude AI interface
    anthropic_interface = AnthropicModelInterface()

    # Clear screen and announce startup
    print("\033[H\033[J", end="")  # ANSI escape codes to clear terminal
    print(f"{AGENT_NAME} has joined")

    # Say hello
    phonemes, _ = g2p("Hello!")  # Convert text to phonemes
    samples, sample_rate = kokoro.create(
        phonemes, KOTORO_VOICE, is_phonemes=True)  # Generate audio
    sd.play(samples, sample_rate)  # Play the audio
    sd.wait()  # Wait for playback to complete

    # Initialize PyAudio for microphone access
    p = pyaudio.PyAudio()
    # Debug line to check audio device info (commented out)
    # print(p.get_device_info_by_index(1))
    # exit()

    try:
        # Main conversation loop - continues until interrupted or terminated
        while True:
            heard_text = ""
            print(f"{AGENT_NAME} is listening. Press any key to stop listening")

            # Start recording audio from microphone
            recorder.start()
            input("")  # Wait for user to press any key to stop recording
            recorder.stop()
            heard_text = recorder.text()  # Get transcribed text

            # Handle case where no speech was detected
            if (not heard_text):
                # Generate and play error message
                phonemes, _ = g2p("It seems you didn't say anything.")
                samples, sample_rate = kokoro.create(
                    phonemes, KOTORO_VOICE, is_phonemes=True)
                sd.play(samples, sample_rate)
                sd.wait()
                continue  # Skip to next iteration

            # Process the user's speech
            PrintUser(heard_text)  # Display what user said
            anthropic_interface.AddUserMessage(
                heard_text)  # Add to conversation history
            response = anthropic_interface.MakeAnthropicAPICall()  # Get AI response
            anthropic_interface.AddAgentMessage(
                response)  # Add AI response to history

            # Process the AI response for file creation commands
            text_to_say = (HandleFileCreationString(response))
            PrintAgent(text_to_say)  # Display cleaned response

            # Generate and play speech if there's text to say
            if (text_to_say):
                phonemes, _ = g2p(text_to_say)
                samples, sample_rate = kokoro.create(
                    phonemes, KOTORO_VOICE, is_phonemes=True)  # Generate audio
                sd.play(samples, sample_rate)  # Play the audio
                sd.wait()  # Wait for completion

                # Check if AI wants to end the conversation
                if CheckForTerminate(text_to_say):
                    exit()  # Terminate program

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        exit()
