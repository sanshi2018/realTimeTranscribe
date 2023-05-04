from typing import Dict

import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.pipelines.utils import hook
from pyannote.core import Segment
from pyannote.core.annotation import Annotation
import torch
from speechbrain.pretrained import EncoderClassifier
from torch import classes

# 音频采样率
SAMPLE_RATE = 16000
# 提取声纹：使用的音频长度
EMBEDDING_DURATION = 1.0
# 提取声纹：只使用达到一定长度的音频
EMBEDDING_DURATION_THRESHOLD = EMBEDDING_DURATION + 0.2
# 提取声纹：读取音频帧数量(torchaaudio)
EMBEDDING_NUM_FRAMES = int(SAMPLE_RATE * EMBEDDING_DURATION)
# 提取声纹：验证声纹达标阈值
EMBENDING_THRESHOLD = 0.25

AUTH_TOKEN = "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"

pipeline: SpeakerDiarization = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1", use_auth_token=AUTH_TOKEN)

output: Annotation = pipeline("ttt.wav")


def load_audio(wav_file_path, frame_offset, num_frames):
    signal, sr = torchaudio.load(str(wav_file_path),
                                 frame_offset=frame_offset,
                                 num_frames=num_frames,
                                 channels_first=False)
    return signal, sr

# 直接使用创建好的分类器
classifier: EncoderClassifier = pipeline._embedding.classifier_

def extract_embedding(classifier: EncoderClassifier, signal, sr):
    waveform = classifier.audio_normalizer(signal, sr)
    # Fake batches:
    batch = waveform.unsqueeze(0)
    # Encode
    emb = classifier.encode_batch(batch, normalize=True)
    return emb


speaker_count = 0

def generate_speaker_name():
    global speaker_count
    speaker_count += 1
    return f"发音人_{speaker_count:02d}"

similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

def speaker_verification(emb1, em2):
    score = 0
    score = float(similarity(emb1, em2)[0])
    decision = score > EMBENDING_THRESHOLD
    return score, decision

def speaker_identification(emb: torch.Tensor, speakers: Dict, segment: Segment):
    result_decision = False
    result_score = 0.0
    result_name = None
    for name, embedding in speakers.items():
        score, decision = speaker_verification(emb, embedding)
        if decision:
            result_decision = True
            result_score = score
            result_name = name
            break

    if not result_decision:
        result_name = generate_speaker_name()

    # 如果不存在则新增，存在则更新。
    speakers[result_name] = emb
    return result_decision, result_score, result_name


tracks = list(output.itertracks())
speakers = {}

for idx in range(0, len(tracks)):
    track, speaker = tracks[idx]
    start = max(track.start, tracks[idx - 1]
                [0].end) if idx != 0 else track.start
    end = min(track.end, tracks[idx + 1]
              [0].start) if idx != len(tracks) - 1 else track.end
    segment = Segment(start=start, end=end)

    print(
        f"{speaker} {track} ==>> start={start:.3f} stop={end:.3f} duration={(end - start):.3f}")

    if segment.duration < EMBEDDING_DURATION_THRESHOLD:
        #print(f"Ignore: start={start:.3f} stop={end:.3f}")
        pass
    else:
        # 加载音频
        frame_offset = int(
            (segment.middle - EMBEDDING_DURATION / 2) * SAMPLE_RATE)
        signal, sr = load_audio("ttt.wav", frame_offset,
                                EMBEDDING_NUM_FRAMES)

        # 提取声纹
        emb = extract_embedding(classifier, signal, sr)
        # 打印出声纹

        # 说话人辨认
        decision, score, name = speaker_identification(emb, speakers, segment)
        print(f"{segment} {decision}, {score:.3f}, {name}")

print(f"说话人数量：{len(speakers)}")
for name, _ in speakers.items():
    print(name)