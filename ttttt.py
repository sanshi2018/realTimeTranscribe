#!/usr/bin/env python3
import datetime
import enum
import logging
from dataclasses import dataclass
from typing import Optional

# import streamSlicer
import faster_whisper
import sounddevice
import whisper, os
import numpy as np
import sounddevice as sd
import threading
# import ringBuffer
import ipdb
from sounddevice import PortAudioError
# from faster_whisper import WhisperModel
from threading import Thread
from scipy.io.wavfile import write

Model = 'tiny'  # Whisper model size (tiny, base, small, medium, large)
English = True  # Use English-only model?
Translate = False  # Translate non-English to English?
SampleRate = 16000  # Stream device recording frequency
BlockSize = 30  # Block size in milliseconds
Threshold = 0.1  # Minimum volume threshold to activate listening
Vocals = [50, 1000]  # Frequency range to detect sounds that could be speech
EndBlocks = 40  # Number of blocks to wait before sending to Whisper

frame_duration = 0.1 # 处理的精度（单位：秒），即对于直播流会切成多长的单位来进行人声判断，不能小于0.03（30ms）
continuous_no_speech_threshold = 0.8 # 连续多少秒没有人声则进行切片
min_audio_length = 3.0 # 切片最小长度
max_audio_length = 30.0 # 切片最大长度
vad_threshold = 0 # 人声判断的阈值，取值0~1，如果你觉得输出吞掉了一些人声，那么可以尝试减少这个值，反之亦然


class Task(enum.Enum):
    TRANSLATE = "translate"
    TRANSCRIBE = "transcribe"


class StreamHandler:
    is_running = None

    def __init__(self, input_device_index: Optional[int], sample_rate: int) -> None:
        # self.history_audio_buffer = ringBuffer.RingBuffer(0 + 1)
        # new
        self.buffer= np.zeros(0, dtype=np.float32)
        self.input_device_index = input_device_index
        self.sample_rate = sample_rate
        self.n_batch_samples = 1 * self.sample_rate  # every  seconds
        self.max_queue_size = 3 * self.n_batch_samples
        self.mutex = threading.Lock()
        self.queue = np.ndarray([], dtype=np.float32)
        # self.stream_slicer = streamSlicer.StreamSlicer(frame_duration=frame_duration,
        #                                                continuous_no_speech_threshold=continuous_no_speech_threshold,
        #                                                min_audio_length=min_audio_length,
        #                                                max_audio_length=max_audio_length,
        #                                                vad_threshold=vad_threshold,
        #                                                sampling_rate=sample_rate)
        # new end
        print("\033[96mLoading Whisper Model..\033[0m", end='', flush=True)
        # self.model = whisper.load_model(f'{Model}{".en" if English else ""}')
        self.model = faster_whisper.WhisperModel(Model, device="cpu", compute_type="float32")
        print("\033[90m Done.\033[0m")

    @staticmethod
    def get_device_sample_rate(device_id: Optional[int]) -> int:
        """Returns the sample rate to be used for recording. It uses the default sample rate
        provided by Whisper if the microphone supports it, or else it uses the device's default
        sample rate.
        """
        whisper_sample_rate = whisper.audio.SAMPLE_RATE
        try:
            sounddevice.check_input_settings(
                device=device_id, samplerate=whisper_sample_rate)
            return whisper_sample_rate
        except PortAudioError:
            device_info = sounddevice.query_devices(device=device_id)
            if isinstance(device_info, dict):
                return int(device_info.get('default_samplerate', whisper_sample_rate))
            return whisper_sample_rate

    def stream_callback(self, in_data: np.ndarray, frame_count, time_info, status):

        # Try to enqueue the next block. If the queue is already full, drop the block.
        chunk: np.ndarray = in_data.ravel()
        with self.mutex:
            if self.queue.size < self.max_queue_size:
                self.queue = np.append(self.queue, chunk)

    @staticmethod
    def amplitude(arr: np.ndarray):
        return (abs(max(arr)) + abs(min(arr))) / 2

    def start(self, input_device_index=1):
        self.is_running = True
        try:
            with sounddevice.InputStream(samplerate=self.sample_rate,
                                         dtype="float32",
                                         channels=1,
                                         callback=self.stream_callback):
                while self.is_running:
                    self.mutex.acquire()
                    if self.queue.size >= self.n_batch_samples:
                        # Dequeue the next block.
                        samples = self.queue[:self.n_batch_samples]
                        # Remove the dequeued block from the queue.
                        self.queue = self.queue[self.n_batch_samples:]
                        self.mutex.release()
                        # Process the block.
                        # 将samples追加写入wav文件
                        self.buffer = np.concatenate((self.buffer, samples))
                        write("test1.wav", self.sample_rate,  self.buffer)
                        # 打印self.buffer的长度
                        # print("len-------------------------"+len(self.buffer))
                        # 打印当前时间
                        # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                        segments, info = self.model.transcribe(samples, beam_size=5, vad_filter=True, language="en")
                        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

                        for segment in segments:
                            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                        # self.stream_slicer.put(audio)
                        # if self.stream_slicer.should_slice():
                        #     # decode audio
                        #     sliced_audio = self.stream_slicer.slice()
                        #     segments, info = self.model.transcribe(audio=sliced_audio, beam_size=5)
                        #     print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
                        #
                        #     for segment in segments:
                        #         print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                    else:
                        self.mutex.release()
        except PortAudioError as exc:
            print(exc)
            self.is_running = False
            logging.exception(exc)
            return


def checkDevice():
    # 获取所有可用的音频设备信息
    print(sd.query_devices())
    # 获取当前默认输入设备
    print(sd.default.device)


def main():
    try:
        device_sample_rate = StreamHandler.get_device_sample_rate(1)
        handler = StreamHandler(input_device_index=1,
                                sample_rate=device_sample_rate)
        handler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print("\n\033[93mQuitting..\033[0m")
        if os.path.exists('dictate.wav'): os.remove('dictate.wav')


if __name__ == '__main__':
    checkDevice()
    main()  # by Nik
