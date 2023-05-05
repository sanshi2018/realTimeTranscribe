#!/usr/bin/env python3
import datetime
import enum
import logging
from dataclasses import dataclass

import faster_whisper
# from faster_whisper import WhisperModel
from typing import Optional

import sounddevice
import whisper, os
import numpy as np
import sounddevice as sd
import threading
import ipdb
from sounddevice import PortAudioError
from threading import Thread
from scipy.io.wavfile import write
import rx.operators as ops

Model = 'tiny'  # Whisper model size (tiny, base, small, medium, large)
English = True  # Use English-only model?
Translate = False  # Translate non-English to English?
SampleRate = 16000  # Stream device recording frequency
BlockSize = 30  # Block size in milliseconds
Threshold = 0.1  # Minimum volume threshold to activate listening
Vocals = [50, 1000]  # Frequency range to detect sounds that could be speech
EndBlocks = 40  # Number of blocks to wait before sending to Whisper


class Task(enum.Enum):
    TRANSLATE = "translate"
    TRANSCRIBE = "transcribe"


class StreamHandler:
    is_running = None

    def __init__(self, input_device_index: Optional[int], sample_rate: int) -> None:
        self.buffer = np.zeros(0, dtype=np.float32)
        # new
        self.input_device_index = input_device_index
        self.sample_rate = sample_rate
        self.n_batch_samples = 1 * self.sample_rate  # every 5 seconds
        self.max_queue_size = 3 * self.n_batch_samples
        self.mutex = threading.Lock()
        self.queue = np.ndarray([], dtype=np.float32)
        self.preHistory=""
        # new end
        print("\033[96mLoading Whisper Model..\033[0m", end='', flush=True)
        # if torch.cuda.is_available() else "cpu"
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

    # def stream_callback(self, in_data: np.ndarray, frame_count, time_info, status):
    def stream_callback(self, indata, frames, time, status):
        # Try to enqueue the next block. If the queue is already full, drop the block.
        # pdb.set_trace()
        # breakpoint()

        # detect is speech
        freq = np.argmax(np.abs(np.fft.rfft(indata[:, 0]))) * self.sample_rate / frames
        if freq < Vocals[0] or freq > Vocals[1]:
            print("not speech")
            return
        # numpy.ndarray to stream
        indata[:].pipe(
            # Ignore this chunk if it does not contain speech
            ops.filter(lambda ann_wav: ann_wav[0].get_timeline().duration() > 0),
        ).subscribe(
            on_next=print("on_next")
        )
        chunk: np.ndarray = indata
        with self.mutex:
            if self.queue.size < self.max_queue_size:
                self.queue = np.append(self.queue, chunk)

    @staticmethod
    def amplitude(arr: np.ndarray):
        return (abs(max(arr)) + abs(min(arr))) / 2

    def start(self, input_device_index=1):
        self.is_running = True
        # segment, info = self.model.transcribe("./chushibiao.mp4", beam_size=5, vad_filter=True)
        # print("language '%s' with probalility %f" % (info.language, info.language_probability))
        #
        # for seg in segment:
        #     print("[%.2fs -> %.2fs] %s" % (seg.start, seg.end, seg.text))
        # return
        try:
            with sounddevice.InputStream(samplerate=self.sample_rate,
                                         device=self.input_device_index,
                                         blocksize=int(self.sample_rate * BlockSize / 1000),
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

                        # audio writ to file
                        # self.buffer = np.concatenate((self.buffer, samples))
                        # write("test.wav", self.sample_rate, self.buffer)

                        # 打印当前时间 时分秒
                        # print("cur Time" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        segment, info = self.model.transcribe(
                            samples, beam_size=5, vad_filter=True, language='en',initial_prompt=self.preHistory)
                        # print("language '%s' with probalility %f time " % (info.language, info.language_probability))
                        for seg in segment:
                            print("[%.2fs -> %.2fs] %s" % (seg.start, seg.end, seg.text))
                            self.preHistory += seg.text
                        if len(self.preHistory) > 100:
                            self.preHistory = ""

                        # result = self.model.transcribe(
                        #     fp16=False,
                        #     audio=samples, language='en' if English else '',
                        #     task=Task.TRANSCRIBE.value)
                        # res = result.get('text')
                        # print("转义结果: %s", res)
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
        device_sample_rate = StreamHandler.get_device_sample_rate(2)
        handler = StreamHandler(input_device_index=2,
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
