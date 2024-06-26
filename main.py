import cv2
import numpy as np
import pyaudio

VOLUME = 1.0  # Volume control (200%)
LABEL_WIDTH = 40  # Width of the label image
LABEL_HEIGHT = 40  # Height of the label image

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

            stream.write(out_data.tobytes(), CHUNK)

            fft_data = np.abs(np.fft.rfft(out_data)) * 2 / CHUNK
            fft_data = fft_data[idx_min:idx_max]

            # Normalize FFT data logarithmically to fit in the image height
            fft_normalized = np.log1p(fft_data) / np.log1p(np.max(fft_data))
            bar_width = img_width // len(fft_normalized)

            # Render the spectrum
            img.fill(0)  # Clear image by filling it with black
            for i, value in enumerate(fft_normalized):
                # Calculate the bar's height and draw it
                bar_height = int(value * img_height)
                start_x = i * bar_width
                cv2.rectangle(img, (start_x, img_height - bar_height), (start_x + bar_width - 1, img_height), (255, 255, 255), -1)

            # Add frequency labels rotated
            for i in range(1, 21):  # From 1000 Hz to 20,000 Hz
                freq = 1000 * i
                if freq > 20000:
                    break
                label = f'{freq}'
                x_position = int((freq - 20) / 19980 * img_width) - 30  # Adjust position to scale within image width
                label_img = np.zeros((LABEL_HEIGHT, LABEL_WIDTH, 3), dtype=np.uint8)
                cv2.putText(label_img, label, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                M = cv2.getRotationMatrix2D((20, 20), 90, 0.75)
                label_img = cv2.warpAffine(label_img, M, (LABEL_WIDTH, LABEL_HEIGHT))
                img_start_y = img_height - LABEL_HEIGHT
                img_start_x = x_position - 20
                x_start_clip = max(img_start_x, 0)
                x_end_clip = min(img_start_x + LABEL_WIDTH, img_width)
                y_start_clip = max(img_start_y, 0)
                y_end_clip = min(img_start_y + LABEL_HEIGHT, img_height)
                img_label_clip = label_img[:, (x_start_clip-img_start_x):(x_end_clip-img_start_x)]
                img[y_start_clip:y_end_clip, x_start_clip:x_end_clip] = np.where(img_label_clip > 0, img_label_clip, img[y_start_clip:y_end_clip, x_start_clip:x_end_clip])

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
