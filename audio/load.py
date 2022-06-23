import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

def load_wav(wav_path: str, sample_rate: int) -> np.ndarray:
    try:
        if wav_path.endswith('wav'):
            signal, _ = librosa.load(wav_path, sr=sample_rate)
            return signal

    except ValueError:
        logger.warning('ValueError in {0}'.format(wav_path))
        return None
    except RuntimeError:
        logger.warning('RuntimeError in {0}'.format(wav_path))
        return None
    except IOError:
        logger.warning('IOError in {0}'.format(wav_path))
        return None
