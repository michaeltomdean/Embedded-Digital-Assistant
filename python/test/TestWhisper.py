from faster_whisper import WhisperModel
import time

model_size = 'base.en'

def main():
    model = WhisperModel(model_size, compute_type="int8", cpu_threads=4)

    start_time = time.time()
    segments, info = model.transcribe("python/test/audio/monologue.ogg")

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print(f"{segment.text} ")

    print(f"Took {time.time() - start_time}s")

if __name__ == "__main__":
    main()