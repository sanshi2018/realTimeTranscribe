import sounddevice as sd
import numpy as np
import whisper
import time

# Set up the stream
duration = 1  # seconds
fs = 16000
stream = sd.InputStream(channels=1, blocksize=fs * duration, samplerate=fs)

# Set up the OpenAI API
model = whisper.load_model('tiny.en')

# Start the stream
with stream:
    while True:
        # Read audio data from the stream
        data = stream.read(fs * duration)

        # Convert the audio data to float32 format
        data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Perform speech-to-text conversion
        text = model.predict(data)

        # Print the transcription
        print(text)

        # Wait for the next block of audio data
        time.sleep(duration)
