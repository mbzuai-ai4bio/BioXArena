import json
import os
import sys
import asyncio
from transformers import AutoTokenizer
import tiktoken
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir))
from get_html import fetch_web_content  
from utils.llm_caller import llm_call 

current_dir = os.path.dirname(__file__)
with open(os.path.join(current_dir, '..', '..', '..', '..', 'configs', 'web_agent.json'), 'r') as f:
    tools_config = json.load(f)

USE_PROMPT = tools_config['user_prompt']
USE_LLM = tools_config['USE_MODEL']


def split_chunks(text: str, model: str):
    """
    Robust chunking:
    - Use tiktoken for GPT/Gemini/Vendor2 and as general fallback.
    - Try to load Hugging Face tokenizers only for explicit HF models (e.g. Qwen).
    - On any exception (网络/SSL/加载失败)，退回到基于字符的简单切分，避免抛出错误。
    """
    model_lower = (model or "").lower()
    # defaults
    chunk_token_limit = 120000
    try:
        # Prefer tiktoken for GPT-like or vendor models (no HF download required)
        if "gpt" in model_lower or "gemini" in model_lower or "vendor2" in model_lower:
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
            chunk_token_limit = 120000
        # deepseek has a dedicated tokenizer available locally in some setups
        elif "deepseek" in model_lower:
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base")
            chunk_token_limit = 120000
        # only try to load HF Qwen if model explicitly references qwen
        elif "qwen" in model_lower:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-72B", trust_remote_code=True)
            chunk_token_limit = 30000
        else:
            # Generic fallback to a common tiktoken encoding
            tokenizer = tiktoken.get_encoding("cl100k_base")
            chunk_token_limit = 120000

        # Tokenize and generate chunks
        all_tokens = tokenizer.encode(text)
        chunks = []
        start = 0
        while start < len(all_tokens):
            end = min(start + chunk_token_limit, len(all_tokens))
            chunk_tokens = all_tokens[start:end]
            # Some tokenizer objects (HF) return ints/ids, need to decode accordingly
            try:
                chunk_text = tokenizer.decode(chunk_tokens)
            except Exception:
                # If decode not supported, join ids as bytes fallback (should rarely happen)
                chunk_text = "".join([str(t) for t in chunk_tokens])
            chunks.append(chunk_text)
            start = end
        return chunks

    except Exception as e:
        # likely network/SSL when loading HF tokenizer; fallback to safe character-splitting
        print(f"[WARN] tokenizer load/usage failed ({e}), falling back to character splitter.")
        # choose a conservative chunk size in characters
        chunk_size_chars = 20000
        chunks = [text[i:i+chunk_size_chars] for i in range(0, len(text), chunk_size_chars)]
        return chunks


async def read_html(text, user_prompt, model=None):
    chunks = split_chunks(text, model)
    chunks = chunks[:1]  
    template = USE_PROMPT["search_conclusion"]
    final_prompt = template.format(user=user_prompt, info=chunks[0])
    answer = await llm_call(final_prompt, model)
    return _get_contents(answer)


async def parse_htmlpage(url: str, user_prompt: str = "", llm: str = None):
    try:
        is_fetch, text = await fetch_web_content(url)
        if not is_fetch:
            return {"content":"failed to fetch web content", "urls":[], "score":-1}

        model_to_use = llm if llm else USE_LLM
        try:
            print(f"use {model_to_use} to parse")
            result = await read_html(text, user_prompt, model=model_to_use)
            return result
        except Exception as e:
            USE_LLM = tools_config['BASE_MODEL']
            print(f"origin llm call failed, use {USE_LLM} to parse：{e}")
            result = await read_html(text, user_prompt, model=USE_LLM)
            return result

    except Exception as e:
        print(f"parse failed: {str(e)}")
        return {"content":"failed to parse web content", "urls":[], "score":-1}


def _get_contents(response: str):
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]
        json_str = json_str.replace("\\(", "\\\\(").replace("\\)", "\\\\)")
        try:
            data = json.loads(json_str)
        except Exception as e:
            print(f"\033[91m response parse failed: {str(e)}\033[0m")
            think_end = response.rfind('</think>')
            if think_end != -1:
                return response[think_end + len('</think>'):].strip()
            else:
                return response
        return data
    except Exception as e:
        print(f"parse failed: {str(e)}")
        return {"content":"failed to parse web content", "urls":[], "score":-1}

async def main():
    query = "what is the content of the page"
    url = "https://proceedings.neurips.cc/paper_files/paper/2022"
    results = await parse_htmlpage(url, query, llm="GpuGeek/Qwen3-30B-A3B-Instruct-2507")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
