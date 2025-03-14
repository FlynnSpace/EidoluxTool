import os
from typing import Dict, Optional
from openai import OpenAI
import re
import json

class DeepSeekMindMap:
    def __init__(self):
        """初始化 DeepSeek 对话模块"""
        self.api_key = self._load_config()
        if not self.api_key:
            raise ValueError("未能从配置文件加载 DeepSeek API 密钥")
        
        # 使用 OpenAI SDK 初始化客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
    
    def _load_config(self) -> str:
        """从配置文件加载 API key
        Returns:
            str: API key
        Raises:
            ValueError: 如果配置文件不存在或格式错误
        """
        config_file = "config.json"
        try:
            if not os.path.exists(config_file):
                raise ValueError(f"配置文件 {config_file} 不存在")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            api_key = config.get("deepseek_api_key")
            if not api_key:
                raise ValueError("配置文件中未找到 deepseek_api_key")
                
            return api_key
        except json.JSONDecodeError:
            raise ValueError(f"配置文件 {config_file} 格式错误")
        except Exception as e:
            raise ValueError(f"读取配置文件出错: {str(e)}")

    def _clean_markdown_content(self, content: str) -> str:
        """清理 markdown 内容，删除首尾行
        Args:
            content: 原始 markdown 内容
        Returns:
            str: 清理后的 markdown 内容
        """
        if content is None:
            return ""
            
        # 按行分割内容
        lines = content.strip().split('\n')
        
        # 删除首尾行
        if len(lines) > 2:  # 确保至少有3行内容
            lines = lines[1:-1]
            
        return '\n'.join(lines)

    def generate_mindmap(self, video_text):
        print("开始生成思维导图...")

        with open('mindmapPrompt.txt', 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()

        user_prompt = f"下面是视频的转录文本：{video_text}"

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=4000,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0
            )
            content = response.choices[0].message.content
            # 清理返回的 markdown 内容
            return self._clean_markdown_content(content)
        except Exception as e:
            print(f"生成思维导图时出错: {str(e)}")
            return None

def process_video_to_mindmap(video_text, output_type: str = "text") -> str:
    """处理视频文本并生成思维导图
    Args:
        video_text: 包含转录文本的字典或字符串
        output_type: 输出类型，'text'或'timestamp'
    Returns:
        str: Markdown格式的思维导图
    """
    # 如果报错就是转录文本报错了
    try:
        text_to_analyze = video_text.get(output_type, "")
        # 获取视频标题（如果存在）
        video_title = video_text.get('title', 'mindmap')
    except:
        return video_text
    
    # 创建 DeepSeek 实例并生成思维导图
    deepseek = DeepSeekMindMap()  # 从配置文件读取 API key
    mindmap = deepseek.generate_mindmap(text_to_analyze)
    
    return mindmap, video_title

if __name__ == "__main__":
    # 示例用法
    from GetVideo import process_urls
    
    # 视频处理设置
    url = "https://www.bilibili.com/video/BV1zPPGeYEr9"
    SEGMENT_LENGTH = 0.8
    LANGUAGE = 'zh'
    OUTPUT_TYPE = "text"
    MODEL_SIZE = 'base'

    # 处理视频获取文本
    result = process_urls(
        url, 
        SEGMENT_LENGTH,
        LANGUAGE,
        model_size=MODEL_SIZE,
        output_type=OUTPUT_TYPE
    )
    
    # 生成思维导图
    mindmap, video_title = process_video_to_mindmap(result, OUTPUT_TYPE)
    
    # 清理文件名（移除不合法字符）
    safe_title = re.sub(r'[\\/:*?"<>|]', '_', video_title)
    
    # 保存思维导图到文件
    filename = f"./outputMindmap/{safe_title}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(mindmap)
    
    print(f"思维导图已生成并保存到 {filename}") 