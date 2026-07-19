"""Expert Heads for MusicUnderstand"""

from .emotion_head import EmotionHead, compute_delta_var
from .language_decoder import LanguageDecoder
from .genre_head import GenreHead, GENRES, NUM_GENRES, GENRE_TO_IDX, IDX_TO_GENRE
from .instrument_head import InstrumentHead, INSTRUMENTS, NUM_INSTRUMENTS, INSTR_TO_IDX, IDX_TO_INSTR

__all__ = [
    "EmotionHead",
    "compute_delta_var",
    "LanguageDecoder",
    "GenreHead",
    "GENRES", "NUM_GENRES", "GENRE_TO_IDX", "IDX_TO_GENRE",
    "InstrumentHead",
    "INSTRUMENTS", "NUM_INSTRUMENTS", "INSTR_TO_IDX", "IDX_TO_INSTR",
]
