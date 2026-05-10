"""Utility functions for code extraction and file management in Kaggle experiments.

Provides helpers to extract code from markdown blocks, manage submission file naming,
and save code to disk.
"""

from pathlib import Path
import re

def save_code_to_file(directory, filename, code_content):
    """Save code content to a file in the specified directory.

    Args:
        directory: Target directory path
        filename: Name of the file to create
        code_content: Code content to write
    """
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = target_dir / filename

    file_path.write_text(code_content, encoding='utf-8')
    
    print(f"文件已成功保存至: {file_path}")


def replace_submission_name (code, _id):
    """Replace submission file names in code with a unique submission file name.

    Args:
        code: The code string to modify
        _id: Unique identifier to use in the submission file name

    Returns:
        Modified code string with updated submission file names
    """
    submission_file_name = f"submission_{_id}.csv"
    modified_code = code
    if "submission/submission.csv" in code:
        modified_code = code.replace("submission/submission.csv", f"submission/{submission_file_name}")
    if "/submission.csv" in modified_code:
        modified_code = modified_code.replace("/submission.csv", f"/{submission_file_name}")

    if "to_csv('submission.csv" in modified_code:
        modified_code = modified_code.replace("to_csv('submission.csv", f"to_csv('submission/{submission_file_name}")
    if 'to_csv("submission.csv' in modified_code:
        modified_code = modified_code.replace('to_csv("submission.csv', f'to_csv("submission/{submission_file_name}')

    if '"submission.csv"' in modified_code:
        modified_code = modified_code.replace('"submission.csv"', f'"{submission_file_name}"')
    if "'submission.csv'" in modified_code:
        modified_code = modified_code.replace("'submission.csv'", f"'{submission_file_name}'")
    
    return modified_code

def read_code(value: str, _id: str) -> str:
    """Extract code if value contains a markdown code block; otherwise return original."""
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", value, re.DOTALL)
    if match:
        value = match.group(1).strip()
    return replace_submission_name(value, _id), value




