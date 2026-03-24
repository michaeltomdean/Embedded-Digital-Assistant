from typing import Iterable
from faster_whisper import WhisperModel
from numpy import ndarray
from sympy import Segment
import os

MODEL_SIZE = "base.en"
CPU_THREADS = os.cpu_count()

class WhisperProccessor:
    """A WhisperProccessor classed designed to use a built in whisper model
    to return a transcript
    """
    
    def __init__(self):
        self.model = WhisperModel(MODEL_SIZE, compute_type="int8", cpu_threads=CPU_THREADS)

    def transcribe(self, audio: ndarray):
        """Transcribe an an audio array

        Args:
            audio (ndarray): Numpy array of audio data

        Returns:
            _type_: _description_
        """
        segments, info = self.model.transcribe(audio)
        return self.__make_transcript(segments,info)

    def __make_transcript(self, segments: Iterable[Segment]):
        text = [segment.text for segment in segments]
        return " ".join(text)