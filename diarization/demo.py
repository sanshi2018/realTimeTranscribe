# 导入pyannote.audio包
from pyannote.audio import Inference
from huggingface_hub import login
from pyannote.audio import Pipeline

login(token="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")

# -----------------------说话人检测
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="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")


# apply the pipeline to an audio file
diarization = pipeline("ttt.wav")

# dump the diarization output to disk using RTTM format
# with open("audio.rttm", "w") as rttm:
#     diarization.write_rttm(rttm)

print(diarization)


# -----------------------语音活动检测
# from pyannote.audio import Model
# model = Model.from_pretrained("pyannote/segmentation",
#                               use_auth_token="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")
#
# from pyannote.audio.pipelines import VoiceActivityDetection
# pipeline = VoiceActivityDetection(segmentation=model)
# HYPER_PARAMETERS = {
#   # onset/offset activation thresholds
#   "onset": 0.5, "offset": 0.5,
#   # remove speech regions shorter than that many seconds.
#   "min_duration_on": 0.0,
#   # fill non-speech regions shorter than that many seconds.
#   "min_duration_off": 0.0
# }
# pipeline.instantiate(HYPER_PARAMETERS)
# vad = pipeline("ttt.wav")
# # `vad` is a pyannote.core.Annotation instance containing speech regions
# # print vad in console
# print(vad)

# -----------------------Speaker embedding
#
# from pyannote.audio import Model
# model = Model.from_pretrained("pyannote/embedding",
#                               use_auth_token="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")
#
# #
#
# inference = Inference(model, window="whole", device="cpu")
# embedding = inference("ttt.wav")
# print("embedding::: ", embedding)
#
#
#
# from pyannote.audio import Inference
# inference = Inference(model, window="sliding",
#                       duration=3.0, step=1.0)
# embeddings = inference("ttt.wav")
# # `embeddings` is a (N x D) pyannote.core.SlidingWindowFeature
# # `embeddings[i]` is the embedding of the ith position of the
# # sliding window, i.e. from [i * step, i * step + duration].
# print("embeddingss::: ", embedding)
#


#------------------------voice-activity-detection
# from pyannote.audio import Pipeline
# pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
#                                     use_auth_token="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")
# output = pipeline("ttt.wav")
#
# for speech in output.get_timeline().support():
#     # active speech between speech.start and speech.end
#     print(speech.start, speech.end)
#     print(speech)


# --------------------------------overlapped-speech-detection

# 1. visit hf.co/pyannote/segmentation and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained overlapped speech detection pipeline
# from pyannote.audio import Pipeline
# pipeline = Pipeline.from_pretrained("pyannote/overlapped-speech-detection",
#                                     use_auth_token="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")
# output = pipeline("ttt.wav")
#
# for speech in output.get_timeline().support():
#     # two or more speakers are active between speech.start and speech.end
#     print(speech)
