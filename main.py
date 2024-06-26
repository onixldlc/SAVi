import cv2
import numpy as np
import pyaudio

VOLUME = 1.0  # Volume control (100%)

# Constants for the audio stream
CHUNK = 1024  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Single channel for microphone
RATE = 44100  # Sample rate (44.1kHz)

def main():
    audio = pyaudio.PyAudio()

    # Open the stream for both recording and playing
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        output=True,
                        frames_per_buffer=CHUNK)

    # Frequency array setup
    freqs = np.fft.rfftfreq(CHUNK, 1/RATE)
    idx_min = np.argmax(freqs > 20)  # First index with frequency > 20 Hz
    idx_max = np.argmax(freqs > 20000)  # First index with frequency > 20,000 Hz
    freqs = freqs[idx_min:idx_max]

    # Create an empty black image
    img_height = 500
    img_width = 1000
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    print("Streaming started. Speak into the microphone. Adjust VOLUME as needed.")

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            in_data = np.frombuffer(data, dtype=np.int16)
            out_data = (in_data * VOLUME).astype(np.int16)
            fft_data = np.abs(np.fft.rfft(out_data)) * 2 / CHUNK
            fft_data = fft_data[idx_min:idx_max]

            # Normalize FFT data to fit in the image height
            fft_normalized = np.clip(fft_data / 250, 0, 1)  # Scale and clip FFT values
            bar_width = img_width // len(fft_normalized)

            # Render the spectrum
            img.fill(0)  # Clear image by filling it with black
            for i, value in enumerate(fft_normalized):
                # Calculate the bar's height and draw it
                bar_height = int(value * img_height)
                start_x = i * bar_width
                cv2.rectangle(img, (start_x, img_height - bar_height), (start_x + bar_width - 1, img_height), (255, 255, 255), -1)

            # Display the resulting frame
            cv2.imshow('FFT Spectrum', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
                break
    finally:
        # Handle cleanup
        stream.stop_stream()
        stream.close()
        audio.terminate()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

