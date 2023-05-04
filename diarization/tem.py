import numpy as np
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, Segment
from pyannote.core.utils.helper import get_class_by_name
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform

# 创建用于预处理音频的Pipeline
class Preprocessing(Pipeline):
    # 预处理Pipeline的参数
    feature_extraction = Uniform(
        "pyannote.audio.features.FeatureExtraction",
        {"audio": "pyannote.audio.signal.SlidingWindowAudioSignal"},
    )
    feature_normalization = Uniform(
        "pyannote.audio.features.Normalization",
        {"method": "minmax", "warping": False},
    )
    dimensionality_reduction = Uniform(
        "pyannote.audio.features.DimensionalityReduction",
        {"n_components": 64, "whiten": False, "random_state": 1234},
    )

    def __call__(self, waveform):
        # 将音频波形转换为pyannote.audio信号对象
        signal = self.feature_extraction(waveform)
        # 归一化音频特征
        features = self.feature_normalization(signal)
        # 降维处理
        features = self.dimensionality_reduction(features)
        return features

# 创建用于说话人分割的Pipeline
class Segmentation(Pipeline):
    # 分割Pipeline的参数
    model = Uniform(
        "pyannote.audio.models.segmentation.SpeakerChangeDetection",
        {"duration": 0.5, "min_duration": 0.1, "pad": 0.5},
    )
    threshold = Uniform("float", 0.5)

    def __call__(self, features):
        # 使用模型进行分割
        scores = self.model(scores=features)
        # 使用二值化器将得分转换为二进制标签
        binarize = Binarize(offset=self.threshold, log_scale=True)
        labels = binarize.apply(scores, dimension=1)
        # 将标签转换为pyannote.core.Annotation对象
        segmentation = Annotation()
        for segment, label in labels.itertracks(yield_label=True):
            if label == "1":
                segmentation[Segment(*segment)] = "speech"
        return segmentation

# 创建音频处理Pipeline
pipeline = Preprocessing() | Segmentation()

# 加载音频数据
def callback(indata, frames, time, status):
    if status:
        print(status)
    waveform = np.copy(indata[:, 0])
    # 处理音频
    segmentation = pipeline(waveform)
    # 输出分割结果
    print(segmentation)

# 创建输入流对象
import sounddevice as sd
stream = sd.InputStream(callback=callback)

# 开始录音
with stream:
    sd.sleep(10000)