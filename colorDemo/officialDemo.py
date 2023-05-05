import rx.operators as ops
import diart.operators as dops
# from diart.sources import MicrophoneAudioSource
from diart.blocks import SpeakerSegmentation, OverlapAwareSpeakerEmbedding
from diart.sources import FileAudioSource
segmentation = SpeakerSegmentation.from_pyannote("pyannote/segmentation")
embedding = OverlapAwareSpeakerEmbedding.from_pyannote("pyannote/embedding")
sample_rate = segmentation.model.sample_rate
file = FileAudioSource("ttt.wav", sample_rate)
# mic = MicrophoneAudioSource(sample_rate)

# ttt.wav文件 作为pipe的文件源

stream = file.stream.pipe(
    # Reformat stream to 5s duration and 500ms shift
    dops.rearrange_audio_stream(sample_rate=sample_rate),
    ops.map(lambda wav: (wav, segmentation(wav))),
    ops.starmap(embedding)
).subscribe(on_next=lambda emb: print(emb.shape))

file.read()