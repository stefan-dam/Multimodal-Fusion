import numpy as np

from fusion.temperature_scaling import softmax_with_temperature
from fusion.expected_vad import expected_vad
from fusion.uncertainty import entropy
from fusion.fuse_vad import fuse_static, fuse_dynamic

from .sample_data import SAMPLE_DATA, TEXT_VAD_MAP, AUDIO_VAD_MAP


# Pretend these were learned from calibraton
T_TEXT = 1.2
T_AUDIO = 1.1
STATIC_W_TEXT = 0.6


def run_one(sample):

    p_text = softmax_with_temperature(sample["text_logits"], T=T_TEXT)
    p_audio = softmax_with_temperature(sample["audio_logits"], T=T_AUDIO)

    v_text = expected_vad(p_text, sample["text_labels"], TEXT_VAD_MAP)
    v_audio = expected_vad(p_audio, sample["audio_labels"], AUDIO_VAD_MAP)

    ent_text = entropy(p_text)
    ent_audio = entropy(p_audio)

    v_static = fuse_static(v_text, v_audio, w_text=STATIC_W_TEXT)

    v_dyn, w_t, w_a = fuse_dynamic(
        v_text, v_audio,
        ent_text=ent_text, ent_audio=ent_audio,
        var_text=0.0, var_audio=0.0,
        asr_conf=sample.get("asr_conf"),
        snr=sample.get("snr"),
    )

    return {
        "id": sample["id"],
        "vad_text": v_text,
        "vad_audio": v_audio,
        "vad_static": v_static,
        "vad_dynamic": v_dyn,
        "w_text": w_t,
        "w_audio": w_a,
        "ent_text": ent_text,
        "ent_audio": ent_audio,
        "y_vad": sample.get("y_vad"),
    }


def main():
    print(f"Running fusion demo on {len(SAMPLE_DATA)} samples...\n")

    for s in SAMPLE_DATA:
        out = run_one(s)

        print(f"== {out['id']} ==")
        print(f"  ASR conf: {s.get('asr_conf')}, SNR: {s.get('snr')}")
        print(f"  entropy text: {out['ent_text']:.3f} | entropy audio: {out['ent_audio']:.3f}")
        print(f"  weights dyn: w_text={out['w_text']:.3f}, w_audio={out['w_audio']:.3f}")

        print(f"  VAD text   : {np.round(out['vad_text'], 3)}")
        print(f"  VAD audio  : {np.round(out['vad_audio'], 3)}")
        print(f"  VAD static : {np.round(out['vad_static'], 3)}")
        print(f"  VAD dynamic: {np.round(out['vad_dynamic'], 3)}")

        if out["y_vad"] is not None:
            mae_static = np.mean(np.abs(out["vad_static"] - out["y_vad"]))
            mae_dyn = np.mean(np.abs(out["vad_dynamic"] - out["y_vad"]))
            print(f"  GT y_vad   : {np.round(out['y_vad'], 3)}")
            print(f"  MAE static : {mae_static:.3f} | MAE dynamic: {mae_dyn:.3f}")

        print()

    print("Done.")


if __name__ == "__main__":
    main()
