import re
import os
import subprocess
import torch
from yt_dlp import YoutubeDL
import whisper
from moviepy import AudioFileClip

def check_ffmpeg():
    """检查 FFmpeg 是否已安装"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def identify_source(url):
    """识别视频来源"""
    # 本地文件模式
    if os.path.exists(url):
        return "local"
    
    # Bilibili模式
    bilibili_pattern = r'(?:https?://)?(?:www\.)?bilibili\.com/video/(?:av\d+|BV[\w]+)'
    if re.match(bilibili_pattern, url):
        return "bilibili"
    
    # YouTube模式
    youtube_pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+'
    if re.match(youtube_pattern, url):
        return "youtube"
    
    return None

def download_video(url, source_type):
    """下载视频
    """
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg 未安装。请安装 FFmpeg 后重试。")
    
    if source_type in ["youtube", "bilibili"]:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'outtmpl': 'downloaded_audio.%(ext)s'
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return "downloaded_audio.mp3"
    elif source_type == "local":
        try:
            audio = AudioFileClip(url)
            audio_path = "extracted_audio.mp3"
            audio.write_audiofile(audio_path, verbose=False, logger=None)
            audio.close()
            return audio_path
        except Exception as e:
            raise RuntimeError(f"处理本地视频文件时出错: {str(e)}")

def transcribe_audio(audio_path, segment_length=30, language=None):
    """将音频转换为文字
    Args:
        audio_path: 音频文件路径
        segment_length: 语音分段长度（秒），默认30秒
        language: 语言代码，如'zh'为中文，None为自动检测
    """
    model = whisper.load_model("base")
    
    # 转写选项
    options = {
        "task": "transcribe",  # 转写任务
        "language": language,  # 语言设置
        "word_timestamps": True,  # 启用词级时间戳
        "condition_on_previous_text": True,  # 考虑上下文
        "no_speech_threshold": 0.6,  # 静音判断阈值
        "compression_ratio_threshold": 2.4,  # 压缩比阈值
        "logprob_threshold": -1.0,  # 语音概率阈值
        # "best_of": 5,  # 生成多个候选结果并选择最佳
    }
    
    # 转写音频
    result = model.transcribe(
        audio_path,
        **options
    )
    
    # 处理分段和标点
    formatted_text = ""
    current_segment = ""
    last_timestamp = 0
    
    for segment in result["segments"]:
        # 检查是否需要分段（基于时间间隔）
        if segment["start"] - last_timestamp > segment_length:
            if current_segment:
                formatted_text += current_segment.strip() + "。\n\n"
                current_segment = ""
        
        # 添加当前文本，处理标点
        text = segment["text"].strip()
        if text:
            # 如果当前段不为空且上一段没有结束标点，添加逗号
            if current_segment and not current_segment.strip()[-1] in "。，！？":
                current_segment += "，"
            current_segment += text
        
        last_timestamp = segment["end"]
    
    # 添加最后一段
    if current_segment:
        formatted_text += current_segment.strip() + "。"
    
    return formatted_text

def process_video(url, segment_length=5, language=None):
    """主处理函数
    Args:
        url: 视频URL或本地文件路径
        segment_length: 语音分段长度（秒）
        language: 语言代码，如'zh'为中文，None为自动检测
    """
    source_type = identify_source(url)
    if not source_type:
        return "无法识别的URL或文件格式"
    
    try:
        audio_path = download_video(url, source_type)
        text = transcribe_audio(audio_path, segment_length, language)
        
        # 清理临时文件
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return text
    
    except Exception as e:
        return f"处理过程中出错: {str(e)}"

if __name__ == "__main__":
    # 测试示例
    print("Initializing VideoDownload test...")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前使用的CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")


    urls = [
        #"test_video.mp4",  # 本地文件示例
        "https://www.bilibili.com/video/BV1LsRKYpEZJ",  # Bilibili示例
        #"https://www.youtube.com/watch?v=dQw4w9WgXcQ"   # YouTube示例
    ]
    
    # 转写设置
    SEGMENT_LENGTH = 2  # 分段长度（秒）
    LANGUAGE = 'zh'      # 语言设置（'zh'为中文，None为自动检测）
    
    for url in urls:
        print(f"\n处理URL: {url}")
        result = process_video(url, SEGMENT_LENGTH, LANGUAGE)
        print(f"转换结果:\n{result}")
