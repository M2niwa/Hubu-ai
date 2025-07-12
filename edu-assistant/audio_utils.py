# audio_utils.py
import wave
import numpy as np
from pathlib import Path
from datetime import datetime
import random
import pyaudio
import logging
from scipy.signal import butter, filtfilt
from logger import assistant_logger


class AudioRecorder:
    """实时音频录制处理类"""

    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024):
        """
        初始化录音设备
        :param sample_rate: 采样率 (Hz)
        :param channels: 声道数 (1=单声道, 2=立体声)
        :param chunk_size: 每次读取的音频块大小
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.frames = []

    def start_recording(self):
        """开始录音"""
        self.is_recording = True
        self.frames = []

        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._record_callback
        )
        assistant_logger.log_audio_event("录音已开始")

    def _record_callback(self, in_data, frame_count, time_info, status):
        """音频流回调函数"""
        if self.is_recording:
            self.frames.append(in_data)
            return (in_data, pyaudio.paContinue)
        else:
            return (in_data, pyaudio.paComplete)

    def stop_recording(self):
        """停止录音"""
        self.is_recording = False
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        assistant_logger.log_audio_event("录音已停止")
        return b''.join(self.frames)

    def get_audio_devices(self):
        """获取可用音频设备列表"""
        devices = []
        for i in range(self.audio.get_device_count()):
            dev_info = self.audio.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': dev_info['name'],
                    'sample_rate': dev_info['defaultSampleRate']
                })
        return devices

    def terminate(self):
        """释放资源"""
        self.audio.terminate()


class AudioProcessor:
    """音频数据处理工具类"""

    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels

    def generate_filename(self, prefix="audio", extension="wav"):
        """生成唯一的音频文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_id = f"{random.randint(1000, 9999):04x}"
        return f"{prefix}_{timestamp}_{rand_id}.{extension}"

    def save_recording(self, audio_data, output_dir="audio/recordings"):
        """
        保存录音到WAV文件
        :param audio_data: 原始PCM音频数据
        :param output_dir: 输出目录
        :return: 保存的文件路径
        """
        output_path = Path(output_dir) / self.generate_filename()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with wave.open(str(output_path), 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)

            assistant_logger.log_audio_event(
                "音频文件已保存",
                filename=output_path.name
            )
            return output_path

        except Exception as e:
            assistant_logger.log_error(
                f"保存音频失败: {str(e)}",
                context="save_recording"
            )
            return None

    def preprocess_audio(self, audio_data):
        """
        预处理音频数据以优化识别
        :param audio_data: 原始音频数据
        :return: 处理后的音频数据
        """
        # 转换为numpy数组
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # 转换为单声道
        if self.channels > 1:
            audio_array = audio_array.reshape(-1, self.channels)
            audio_array = np.mean(audio_array, axis=1).astype(np.int16)

        # 归一化
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = (audio_array / max_val * 32767).astype(np.int16)

        # 降噪处理
        audio_array = self.apply_noise_reduction(audio_array)

        # 静音修剪
        audio_array = self.trim_silence(audio_array)

        return audio_array.tobytes()

    def apply_noise_reduction(self, audio_array, cutoff=300, order=5):
        """应用高通滤波器去除背景噪声"""
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        filtered = filtfilt(b, a, audio_array.astype(np.float32))
        return filtered.astype(np.int16)

    def trim_silence(self, audio_array, threshold=0.03, min_silence=0.5):
        """
        去除开头和结尾的静音部分
        :param threshold: 静音阈值（振幅百分比）
        :param min_silence: 最小静音持续时间（秒）
        """
        threshold = int(threshold * 32767)
        min_silence_frames = int(min_silence * self.sample_rate)

        # 查找起始点
        start_index = 0
        for i in range(0, len(audio_array) - min_silence_frames, min_silence_frames):
            if np.max(np.abs(audio_array[i:i + min_silence_frames])) > threshold:
                start_index = i
                break

        # 查找结束点
        end_index = len(audio_array)
        for i in range(len(audio_array) - 1, min_silence_frames, -min_silence_frames):
            if np.max(np.abs(audio_array[i - min_silence_frames:i])) > threshold:
                end_index = i
                break

        return audio_array[start_index:end_index]

    def convert_pcm_to_wav(self, pcm_path, output_dir="audio/converted"):
        """
        将PCM文件转换为WAV格式
        :param pcm_path: PCM文件路径
        :param output_dir: 输出目录
        """
        output_path = Path(output_dir) / f"{Path(pcm_path).stem}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(pcm_path, 'rb') as pcm_file:
                pcm_data = pcm_file.read()

            with wave.open(str(output_path), 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm_data)

            assistant_logger.log_audio_event(
                "PCM转WAV完成",
                filename=output_path.name
            )
            return output_path

        except Exception as e:
            assistant_logger.log_error(
                f"PCM转WAV失败: {str(e)}",
                context="convert_pcm_to_wav"
            )
            return None


# 全局音频处理器实例
audio_processor = AudioProcessor()