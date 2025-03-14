import re
import os
import subprocess
import torch
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
from moviepy import AudioFileClip
from opencc import OpenCC
from tqdm import tqdm
from googletrans import Translator
import time
import asyncio

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
    if "bilibili" in url.lower():
        return "bilibili"
    
    # YouTube模式
    if "youtube" in url.lower() in url.lower():
        return "youtube"
    
    return None

def download_video(url, source_type):
    """下载视频"""
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg 未安装。请安装 FFmpeg 后重试。")
    
    if source_type in ["youtube", "bilibili"]:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'outtmpl': 'downloaded_audio.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'force_generic_extractor': False
        }
        
        video_title = None
        
        # 首先获取视频标题
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', None)
        except Exception as e:
            print(f"获取视频标题失败: {str(e)}")
        
        # 首先尝试使用cookies文件
        cookies_file = "youtube.com_cookies.txt"
        if os.path.exists(cookies_file):
            print("使用cookies文件下载...")
            ydl_opts['cookiefile'] = cookies_file
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                return "downloaded_audio.mp3", video_title
            except Exception as e:
                print(f"使用cookies文件下载失败: {str(e)}")
        
        # 如果cookies文件不存在或失败，尝试使用浏览器cookies
        browsers = [
            ('firefox', 'Firefox'),
            ('chrome', 'Chrome'),
            ('edge', 'Edge'),
            ('opera', 'Opera'),
            ('brave', 'Brave')
        ]
        
        for browser, browser_name in browsers:
            try:
                print(f"尝试使用 {browser_name} 浏览器的 cookies...")
                ydl_opts['cookiesfrombrowser'] = (browser,)
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                return "downloaded_audio.mp3", video_title
            except Exception as e:
                print(f"使用 {browser_name} 浏览器的 cookies 失败: {str(e)}")
                continue
        
        # 如果所有方法都失败，尝试不使用cookies
        print("所有cookies方法都失败，尝试不使用cookies下载...")
        try:
            if 'cookiesfrombrowser' in ydl_opts:
                del ydl_opts['cookiesfrombrowser']
            if 'cookiefile' in ydl_opts:
                del ydl_opts['cookiefile']
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return "downloaded_audio.mp3", video_title
        except Exception as e:
            raise RuntimeError(f"下载失败: {str(e)}\n请按照以下步骤操作：\n1. 使用浏览器扩展（如 'Get cookies.txt'）导出YouTube的cookies\n2. 将导出的cookies文件重命名为 'youtube.com_cookies.txt' 并放在程序同目录下\n3. 确保已在浏览器中登录YouTube账号\n4. 如果使用Chrome，请先关闭所有Chrome窗口")
    elif source_type == "local":
        try:
            audio = AudioFileClip(url)
            audio_path = "extracted_audio.mp3"
            audio.write_audiofile(audio_path, verbose=False, logger=None)
            audio.close()
            # 对于本地文件，使用文件名作为标题
            video_title = os.path.splitext(os.path.basename(url))[0]
            return audio_path, video_title
        except Exception as e:
            raise RuntimeError(f"处理本地视频文件时出错: {str(e)}")

def convert_to_simplified(text):
    """将繁体中文转换为简体中文"""
    cc = OpenCC('t2s')  # 繁体转简体
    return cc.convert(text)

def format_timestamp(seconds):
    """将秒数转换为时:分:秒格式"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

async def translate_text(text, src='en', dest='zh-cn'):
    """使用 googletrans 翻译文本"""
    translator = Translator()
    try:
        # 添加重试机制
        max_retries = 3
        for i in range(max_retries):
            try:
                # 将文本分成较小的块进行翻译
                chunk_size = 5000
                translated_chunks = []
                
                # 按字符长度分割文本
                chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
                
                with tqdm(total=len(chunks), desc="翻译进度") as pbar:
                    for chunk in chunks:
                        result = await translator.translate(chunk, src=src, dest=dest)
                        translated_chunks.append(result.text)
                        pbar.update(1)
                
                return "".join(translated_chunks)
            except Exception as e:
                if i == max_retries - 1:
                    raise e
                await asyncio.sleep(1)  # 等待1秒后重试
    except Exception as e:
        print(f"翻译出错: {str(e)}")
        return text  # 如果翻译失败，返回原文

async def translate_content(text, segments, src='en', dest='zh-cn', output_type="both"):
    """翻译所有内容
    Args:
        text: 正文文本
        segments: 时间戳文本列表
        src: 源语言
        dest: 目标语言
        output_type: 输出类型
            - "text": 仅翻译正文
            - "timestamp": 仅翻译时间戳
            - "both": 两者都翻译
    """
    print("\n检测到非中文内容，开始翻译...")
    
    translated_text = text
    translated_segments = segments
    
    if output_type in ["text", "both"]:
        print("翻译正文...")
        translated_text = await translate_text(text, src, dest)
    
    if output_type in ["timestamp", "both"]:
        print("翻译时间戳文本...")
        translated_segments = []
        with tqdm(total=len(segments), desc="翻译时间戳") as pbar:
            for segment in segments:
                translated_text_segment = await translate_text(segment['text'], src, dest)
                translated_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': translated_text_segment
                })
                pbar.update(1)
    
    return translated_text, translated_segments

def transcribe_audio(audio_path, segment_length=30, audio_language=None, model_size="large-v3", compute_precision="float16", num_workers=4, beam_size=5, output_type="both"):
    """将音频转换为文字"""
    # 检查是否可以使用 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 根据设备选择最佳计算精度
    if device == "cpu" and compute_precision == "float16":
        compute_precision = "int8"
    elif device == "cuda" and compute_precision == "int8":
        compute_precision = "float16"
    
    # 初始化模型
    print(f"正在加载模型... (型号: {model_size}, 计算精度: {compute_precision})")
    model = WhisperModel(
        model_size_or_path=model_size,
        device=device,
        compute_type=compute_precision,
        download_root="models",  # 模型下载位置
        num_workers=num_workers,  # 并行处理线程数
        cpu_threads=num_workers   # CPU线程数
    )
    
    # 获取音频时长
    audio = AudioFileClip(audio_path)
    duration = audio.duration
    audio.close()
    
    # 如果未指定语言，先进行语言检测
    if audio_language is None:
        print("\n正在检测语言...")
        segments_generator, info = model.transcribe(
            audio_path,
            beam_size=beam_size,
            language=None,
            without_timestamps=True,
            initial_prompt=None,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400,
            )
        )
        # 将生成器转换为列表
        segments = list(segments_generator)
        detected_language = info.language
        print(f"检测到的语言: {detected_language}")
        audio_language = detected_language
    
    # 转写音频
    print(f"\n转写音频...")
    with tqdm(total=100, desc="转录进度") as pbar:
        segments_generator, _ = model.transcribe(
            audio_path,
            beam_size=beam_size,
            language=audio_language,  # 使用检测到的语言
            task="transcribe",  # 只进行转写
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,  # 最小静音持续时间
                speech_pad_ms=400,  # 语音片段填充
            ),
            initial_prompt=None,
            condition_on_previous_text=True,
            temperature=0.0  # 使用确定性解码
        )
        segments = []
        current_progress = 0
        for segment in segments_generator:
            segments.append(segment)
            # 计算当前进度，保留一位小数
            progress = min(100, round(100 * segment.end / duration, 1))
            # 只更新显示实际增加的进度
            if progress > current_progress:
                pbar.update(progress - current_progress)
                current_progress = progress
        # 强制更新到100%
        if current_progress < 100:
            pbar.update(100 - current_progress)
    
    # 处理分段和标点
    formatted_text = ""
    current_segment = ""
    last_timestamp = 0
    
    # 保存带时间戳的文本
    timestamped_segments = []
    
    for segment in segments:
        start_time = segment.start
        end_time = segment.end
        text = segment.text.strip()
        
        # 检查是否需要分段
        if start_time - last_timestamp > segment_length:
            if current_segment:
                formatted_text += current_segment.strip() + "。\n\n"
                current_segment = ""
        

        if text:
            timestamped_segments.append({
                "start": start_time,
                "end": end_time,
                "text": text
            })
            
            if current_segment and not current_segment.strip()[-1] in "。，！？":
                current_segment += "，"
            current_segment += text
        
        last_timestamp = end_time
    
    if current_segment:
        formatted_text += current_segment.strip() + "。"
    
    # 如果检测到的语言不是中文，进行翻译
    if audio_language.lower() not in ['zh', 'chi', 'chinese']:
        print(f"\n检测到非中文内容 ({audio_language})，开始翻译...")
        # 使用异步事件循环进行翻译
        translated_text, translated_segments = asyncio.run(
            translate_content(formatted_text, timestamped_segments, src=audio_language, dest='zh-cn', output_type=output_type)
        )
        formatted_text = translated_text
        timestamped_segments = translated_segments
        print("翻译完成！")
    
    # 生成带时间戳的完整文本
    timestamped_text = "转录文本（带时间戳）：\n"
    for segment in timestamped_segments:
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text_line = f"[{start} --> {end}] {segment['text']}"
        timestamped_text += text_line + "\n\n"
    
    # 返回处理结果
    return {
        "text": formatted_text,  # 中文正文
        "timestamped_text": timestamped_text,  # 带时间戳的中文文本
        "title": None
    }

def process_video(url, segment_length=5, language=None, model_size="large-v3", compute_precision="float16", num_workers=4, beam_size=5, output_type="both"):
    """主处理函数
    Args:
        url: 视频URL或本地文件路径
        segment_length: 语音分段长度（秒）
        language: 语言代码，如'zh'为中文，None为自动检测
        model_size: 模型大小，可选值：
            - "tiny": ~39M参数
            - "base": ~74M参数
            - "small": ~244M参数
            - "medium": ~769M参数
            - "large-v3": ~1.5B参数
        compute_precision: 计算精度，可选值：
            - "float16": 半精度浮点（更快，需要GPU）
            - "int8": 8位整数（CPU推理的最佳选择）
            - "float32": 单精度浮点（更准确但更慢）
        num_workers: 并行处理的工作线程数
        beam_size: 波束搜索大小，更大的值可能提高准确性但会降低速度
        output_type: 输出类型
            - "text": 仅翻译正文
            - "timestamp": 仅翻译时间戳
            - "both": 两者都翻译
    """
    source_type = identify_source(url)
    if not source_type:
        return "无法识别的URL或文件格式"
    
    try:
        audio_path, video_title = download_video(url, source_type)
        result = transcribe_audio(
            audio_path, 
            segment_length, 
            language,
            model_size=model_size,
            compute_precision=compute_precision,
            num_workers=num_workers,
            beam_size=beam_size,
            output_type=output_type
        )
        
        # 添加视频标题到结果中
        if isinstance(result, dict):
            result['title'] = video_title
        
        # 清理临时文件
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return result
    
    except Exception as e:
        return f"处理过程中出错: {str(e)}"

def check_cuda():
    """检查CUDA状态并返回信息"""
    cuda_info = {
        "available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "version": torch.version.cuda if torch.cuda.is_available() else None
    }
    return cuda_info

def process_urls(url, segment_length=2, language=None, print_info=True, 
                model_size="large-v3", compute_precision="float16", num_workers=4, beam_size=5, output_type="both"):
    """处理单个URL的主函数
    Args:
        url: 视频URL或本地文件路径
        segment_length: 语音分段长度（秒），默认2秒
        language: 语言代码，如'zh'为中文，None为自动检测
        print_info: 是否打印处理信息，默认True
        model_size: 模型大小，可选值：
            - "tiny": ~39M参数
            - "base": ~74M参数
            - "small": ~244M参数
            - "medium": ~769M参数
            - "large-v3": ~1.5B参数
        compute_precision: 计算精度，可选值：
            - "float16": 半精度浮点（更快，需要GPU）
            - "int8": 8位整数（CPU推理的最佳选择）
            - "float32": 单精度浮点（更准确但更慢）
        num_workers: 并行处理的工作线程数
        beam_size: 波束搜索大小，更大的值可能提高准确性但会降低速度
        output_type: 输出类型
            - "text": 仅翻译正文
            - "timestamp": 仅翻译时间戳
            - "both": 两者都翻译
    Returns:
        dict: 包含以下字段的字典：
            - text (str): 连续的纯文本，已处理分段和标点，适合阅读
            - timestamped_text (str): 带时间戳的文本，格式：[00:00:00.00 --> 00:00:05.00] 文本内容
    """
    # 检查CUDA状态
    cuda_info = check_cuda()
    if print_info:
        print("初始化视频处理...")
        print(f"CUDA是否可用: {cuda_info['available']}")
        if cuda_info['available']:
            print(f"当前使用的CUDA设备: {cuda_info['device_name']}")
            print(f"CUDA版本: {cuda_info['version']}")
        print(f"模型配置: {model_size}, 计算精度: {compute_precision}, 线程数: {num_workers}")
        print(f"\n处理URL: {url}")
    
    result = process_video(
        url, 
        segment_length, 
        language, 
        model_size=model_size,
        compute_precision=compute_precision,
        num_workers=num_workers,
        beam_size=beam_size,
        output_type=output_type
    )
    
    return result


# TODO: 时间戳的翻译功能是逐条提交的，非常耗时，并且翻译后会和text合并
if __name__ == "__main__":
    #url = "test_video.mp4",  # 本地文件示例
    #url = "https://www.bilibili.com/video/BV1NJ9QY8E6c",  # Bilibili示例
    url = "https://www.youtube.com/watch?v=LP5OCa20Zpg"   # YouTube示例
    
    # 转写设置
    SEGMENT_LENGTH = 2  # 分段长度（秒）
    LANGUAGE = None     # 语言设置（'zh'为中文，None为自动检测，耗时）
    OUTPUT_TYPE = "text"  # 输出类型：'text'仅纯文本，'timestamp'仅时间戳
    
    # 性能设置
    MODEL_SIZE = "base"  # 使用中等大小的模型以平衡速度和准确性
    COMPUTE_PRECISION = "float16"  # GPU使用float16，CPU会自动切换到int8
    NUM_WORKERS = 4  # 并行处理线程数
    BEAM_SIZE = 3  # 较小的波束搜索大小以提高速度
    
    # 使用封装的函数处理
    result = process_urls(
        url, 
        SEGMENT_LENGTH, 
        LANGUAGE, 
        model_size=MODEL_SIZE,
        compute_precision=COMPUTE_PRECISION,
        num_workers=NUM_WORKERS,
        beam_size=BEAM_SIZE,
        output_type = OUTPUT_TYPE,
    )

    print(result[OUTPUT_TYPE])