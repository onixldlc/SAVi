import cv2
import numpy as np
import pyaudio

VOLUME = 1.0  # Volume control (200%)
RATE = 44100  # Sample rate (44.1kHz)
CHUNK = 8192  # Number of audio samples per frame, increased for better frequency resolution

def main():
    audio = pyaudio.PyAudio()

    # Open the stream for both recording and playing
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=RATE,
                        input=True,
                        output=True,
                        frames_per_buffer=CHUNK)

    # Frequency array setup
    freqs = np.fft.rfftfreq(CHUNK, 1 / RATE)
    idx_min = np.argmax(freqs > 60)  # First index with frequency > 1 Hz
    idx_max = np.argmax(freqs >= 1400)  # Last index with frequency <= 1000 Hz
    if idx_max == 0:  # If 1000 Hz is beyond the frequency range in our array, use the maximum index
        idx_max = len(freqs) - 1

    # Create an empty black image
    img_height = 750
    img_width = 1000
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    print("Streaming started. Speak into the microphone. Adjust VOLUME as needed.")

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            in_data = np.frombuffer(data, dtype=np.int16)
            out_data = (in_data * VOLUME).astype(np.int16)
            stream.write(out_data.tobytes(), CHUNK)

            # Perform FFT and take absolute value
            fft_data = np.abs(np.fft.rfft(out_data)) * 2 / CHUNK
            fft_data = fft_data[idx_min:idx_max+1]

            # Normalize FFT data logarithmically to fit in the image height
            fft_normalized = np.log1p(fft_data) / np.log1p(np.max(fft_data))

            # Render the spectrum
            img.fill(0)  # Clear image by filling it with black
            for i, value in enumerate(fft_normalized):
                # Calculate the bar's height and draw it
                bar_height = int(value * img_height)
                start_x = int((i / len(fft_normalized)) * img_width)  # Scale x position to image width
                cv2.rectangle(img, (start_x, img_height - bar_height), (start_x + 1, img_height), (255, 255, 255), -1)

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
