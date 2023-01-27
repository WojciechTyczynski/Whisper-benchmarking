import jiwer
from whisper.normalizers import EnglishTextNormalizer


def get_WER_MultipleTexts(transcription, reference):
    """Calculate WER between transcription and reference."""
    normalizer = EnglishTextNormalizer()
    transcription = [normalizer(text) for text in transcription]
    reference = [normalizer(text) for text in reference]
    wer = jiwer.wer(reference, transcription)
    return wer

def get_WER_SingleText(transcription, reference):
    """Calculate WER between transcription and reference."""
    normalizer = EnglishTextNormalizer()
    transcription = normalizer(transcription)
    reference = normalizer(reference)
    wer = jiwer.wer(reference, transcription)
    return wer


