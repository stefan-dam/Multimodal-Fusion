import numpy as np

# Emotion label sets

TEXT_LABELS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "neutral",
]

AUDIO_LABELS = [
    "happy",
    "sad",
    "angry",
    "fearful",
    "calm",
]

# Example VAD mappings

TEXT_VAD_MAP = {
    "joy": (0.9, 0.6, 0.6),
    "sadness": (0.2, 0.4, 0.3),
    "anger": (0.1, 0.8, 0.7),
    "fear": (0.2, 0.7, 0.4),
    "surprise": (0.7, 0.8, 0.5),
    "disgust": (0.1, 0.6, 0.6),
    "neutral": (0.5, 0.3, 0.5),
}

AUDIO_VAD_MAP = {
    "happy": (0.85, 0.65, 0.6),
    "sad": (0.25, 0.35, 0.3),
    "angry": (0.15, 0.85, 0.7),
    "fearful": (0.2, 0.75, 0.4),
    "calm": (0.6, 0.2, 0.55),
}

# Utility to generate logits

def random_logits(k, scale=1.5):
    """Generate random logits with mild structure."""
    return np.random.normal(loc=0.0, scale=scale, size=k)


# Sample dataset

SAMPLE_DATA = [
    {
        "id": "sample_1",
        "text_logits": random_logits(len(TEXT_LABELS)),
        "audio_logits": random_logits(len(AUDIO_LABELS)),
        "text_labels": TEXT_LABELS,
        "audio_labels": AUDIO_LABELS,
        "asr_conf": 0.92,      # good transcription
        "snr": 18.0,           # clean audio
        "y_vad": np.array([0.7, 0.5, 0.55]),  # optional ground truth
    },
    {
        "id": "sample_2",
        "text_logits": random_logits(len(TEXT_LABELS)),
        "audio_logits": random_logits(len(AUDIO_LABELS)),
        "text_labels": TEXT_LABELS,
        "audio_labels": AUDIO_LABELS,
        "asr_conf": 0.40,      # poor ASR
        "snr": 16.0,
        "y_vad": np.array([0.3, 0.6, 0.4]),
    },
    {
        "id": "sample_3",
        "text_logits": random_logits(len(TEXT_LABELS)),
        "audio_logits": random_logits(len(AUDIO_LABELS)),
        "text_labels": TEXT_LABELS,
        "audio_labels": AUDIO_LABELS,
        "asr_conf": 0.85,
        "snr": 6.0,            # noisy audio
        "y_vad": np.array([0.4, 0.7, 0.45]),
    },
    {
        "id": "sample_4",
        "text_logits": random_logits(len(TEXT_LABELS)),
        "audio_logits": random_logits(len(AUDIO_LABELS)),
        "text_labels": TEXT_LABELS,
        "audio_labels": AUDIO_LABELS,
        "asr_conf": 0.95,
        "snr": 22.0,
        "y_vad": np.array([0.8, 0.4, 0.6]),
    },
    {
        "id": "sample_5",
        "text_logits": random_logits(len(TEXT_LABELS)),
        "audio_logits": random_logits(len(AUDIO_LABELS)),
        "text_labels": TEXT_LABELS,
        "audio_labels": AUDIO_LABELS,
        "asr_conf": 0.55,
        "snr": 10.0,
        "y_vad": np.array([0.45, 0.55, 0.5]),
    },
]


if __name__ == "__main__":
    
    print(f"Loaded {len(SAMPLE_DATA)} samples\n")

    for s in SAMPLE_DATA:
        print(
            s["id"],
            "| text logits:", s["text_logits"].shape,
            "| audio logits:", s["audio_logits"].shape,
            "| ASR:", s["asr_conf"],
            "| SNR:", s["snr"],
        )
