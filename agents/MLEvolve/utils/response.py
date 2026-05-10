import json
import re

import black


def wrap_code(code: str, lang="python") -> str:
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script):
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_jsons(text):
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass

    if len(json_objects) == 0 and not text.endswith("}"):
        json_objects = extract_jsons(text + "}")
        if len(json_objects) > 0:
            return json_objects

    return json_objects


def trim_long_string(string, threshold=5100, k=2500):
    if len(string) > threshold:
        first_k_chars = string[:k]
        last_k_chars = string[-k:]
        truncated_len = len(string) - 2 * k
        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string


def extract_code(text):
    parsed_codes = []

    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    valid_code_blocks = [
        format_code(c) for c in parsed_codes if is_valid_python_script(c)
    ]
    return format_code("\n\n".join(valid_code_blocks))


def extract_text_up_to_code(s):
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()


def extract_plan_from_diff_response(text: str) -> str:
    if not text:
        return ""

    stop_tokens = [
        "<<<<<<< SEARCH",
        "< SEARCH",
        ">>>>>>> REPLACE",
        "=======",
        "```",
    ]

    def cut_at_stop(s: str) -> str:
        indices = [s.find(token) for token in stop_tokens if s.find(token) != -1]
        if indices:
            return s[: min(indices)]
        return s

    if "Fixed Code Plan:" in text:
        candidate = text.split("Fixed Code Plan:", 1)[1]
        return cut_at_stop(candidate).strip()

    if "Plan:" in text:
        candidate = text.split("Plan:", 1)[1]
        return cut_at_stop(candidate).strip()

    return cut_at_stop(text).strip()


def extract_review(text):
    parsed_codes = []

    matches = re.findall(r"```(json)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(json)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    if len(parsed_codes) == 0 or not parsed_codes[0].strip():
        json_objects = extract_jsons(text)
        if len(json_objects) > 0:
            return json_objects[0]
        raise ValueError(f"No JSON found in text")

    try:
        review = json.loads(parsed_codes[0].strip())
        return review
    except json.JSONDecodeError:
        json_objects = extract_jsons(text)
        if len(json_objects) > 0:
            return json_objects[0]
        raise


def format_code(code) -> str:
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code
