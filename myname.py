from pydoc import text
import time
from typing import Type
from openai import OpenAI
import wave
import threading
import keyboard
import vosk
from vosk import Model, KaldiRecognizer
import os
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
import pyttsx3
import sys
import queue
import sounddevice as sd
import json
import pyaudio
import tkinter as tk
#语音识别
def speech_to_text(wav_file):
    #模型初始化  
    model = Model(r"/models/vosk-model-cn-0.22")
    wf = wave.open(wav_file, "rb")
   
    rec = KaldiRecognizer(model, wf.getframerate())
    #音频流识别
    while True:
        data = wf.readframes(16000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            rec.Result()
    return rec.FinalResult()
    #返回识别结果

#deepseek
def chat(text):
    #身份验证
    client = OpenAI(api_key="sk-4a541283c8034a48abb57bb721a30e57", base_url="https://api.deepseek.com")
    #发送请求
    response = client.chat.completions.create(
    model="deepseek-chat",#模型选择
    messages=[
        {"role": "system", "content": "请以口语化的风格回答问题"},#测试文本
        {"role": "user", "content": text},
    ],
    stream=False
    )
    return response.choices[0].message.content

def printchat(text,delay=0.1):#逐字打印,仅确认chat模型函数运行
    for char in text:
        print (char, end="", flush=True)
        time.sleep(delay)
    print()
    
#文本转语音
def text_to_speech(text):    
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)#设置语速
    engine.say(text)
    engine.runAndWait()
    engine.stop()

#第三代
CHUNK = 1024
FORMAT = pyaudio.paInt16  # PCM格式
CHANNELS = 1              # 单声道
RATE = 16000              # 采样率
RECORD_SECONDS = 20        # 默认录音时长(如果不用'b'停止)
FILENAME = "realtime.wav"

class AudioRecorder:#录音类
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.stream = None

    def save_recording(self):
        # 保存为WAV文件
        wf = wave.open(FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print(f"音频已保存为 {FILENAME}")

    def start_recording(self):
        self.frames = []
        self.is_recording = True
        
        # 打开音频流
        self.stream = self.audio.open(format=FORMAT,
                                    channels=CHANNELS,
                                    rate=RATE,
                                    input=True,
                                    frames_per_buffer=CHUNK)
        
        print("录音开始...")
        while self.is_recording:
            data = self.stream.read(CHUNK)
            self.frames.append(data)
        
        # 录音结束后关闭流
        self.stream.stop_stream()
        self.stream.close()
        self.save_recording()
        
    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            print("录音结束")
    

def main():#主函数
    recorder = AudioRecorder()
    root = tk.Tk()
    root.title("声音录制与识别")
    
    display = tk.Text(root, height=10, width=50)
    display.pack(pady=10)
    def text_process(string):
        return string[12:-1]
    def real_start():
        recording_thread = threading.Thread(target=recorder.start_recording)
        recording_thread.start()
    def the_end():
         if recorder.is_recording:
            recorder.stop_recording()
            display.insert(tk.END, f"{'语音识别中'}\n")
            processed_text = text_process(speech_to_text(FILENAME))
            display.insert(tk.END, f"你说: {processed_text}\n")
            display.update_idletasks()     
            display.yview(tk.END)
            c=chat(text_process(speech_to_text(FILENAME)))
            text_to_speech(c)
            
            display.update_idletasks()
    start_button = tk.Button(root, text="开始录制声音", command=real_start)
    start_button.pack(pady=10)
    stop_button = tk.Button(root, text="停止录制声音", command=the_end)
    stop_button.pack(pady=10)
    root.mainloop()
    recorder.audio.terminate()
    
if __name__ == "__main__":
    main()
    
    
#第二代
# samplerate = 16000  # 采样率
# channels = 1        # 单声道
# dtype = np.int16    # pcm 16位
# filename = "realtime.wav"

# # 全局变量
# audio_buffer = []
# is_recording = False
# def callback(indata, frames, time, status):
#     """实时音频流回调函数"""
#     if is_recording:
#         audio_buffer.append(indata.copy())  # 录制时保存数据
# # 初始化音频流
# stream = sd.InputStream(
#     samplerate=samplerate,
#     channels=channels,
#     dtype=dtype,
#     callback=callback
# )
# stream.start()
# print("按 'a' 开始录音，按 'b' 结束并保存...")
# # 监听键盘输入

# while True:
#     if keyboard.is_pressed('a') and not is_recording:
#         is_recording = True
#         audio_buffer = []  # 清空旧数据
#         print("录音开始...")
    
#     if keyboard.is_pressed('b') and is_recording:
#         is_recording = False
#         print("录音结束，保存文件中...")
        
#         # 拼接数据并保存
#         if audio_buffer:
#             audio_data = np.concatenate(audio_buffer)
#             write(filename, samplerate, audio_data)
#             print(f"已保存为 {filename}")     
#             break  # 退出循环

# stream.stop()
# stream.close()
# a=speech_to_text("realtime.wav")
# print(a)
#第一代
# audio_queue = queue.Queue()    
# def callback(indata, frames, time, status):#实时音频流回调函数:
#     audio_queue.put(indata.copy())

# def real_time_recognition(duration=10):#流式处理，实时语音识别
#     samplerate =16000 #采样率
#     with sd.InputStream(
#         samplerate=samplerate, 
#         channels=1,
#         callback=callback,
#         dtype=np.int16):
#         print(f"录音中...（{duration}秒）,按 Ctrl+C 停止整个程序")
#         sd.sleep(duration * 1000)
    
#     # 拼接音频数据
#     audio_data = []
#     while not audio_queue.empty():
#         audio_data.append(audio_queue.get())
    
#     # 保存为WAV直接处理
#     audio_array = np.concatenate(audio_data)
#     write("realtime.wav", samplerate, audio_array)
#     return speech_to_text("realtime.wav")   
#测试
