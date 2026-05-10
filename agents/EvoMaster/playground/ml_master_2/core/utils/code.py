from pathlib import Path
import re

def save_code_to_file(directory, filename, code_content):
    """Save code content to a file in the specified directory.

    Args:
        directory: Target directory path.
        filename: Name of the file to create.
        code_content: Code content to write.
    """
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / filename

    file_path.write_text(code_content, encoding='utf-8')

    print(f"File successfully saved to: {file_path}")


def replace_submission_name(code, _id):
    """Replace submission.csv references with a unique submission file name.

    Args:
        code: The code string to modify.
        _id: Unique identifier for the submission file.

    Returns:
        Modified code with updated submission file references.
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
    """Extract code from markdown code blocks and replace submission file names.

    Args:
        value: The text potentially containing a markdown code block.
        _id: Unique identifier for the submission file.

    Returns:
        A tuple of (modified_code, original_code).
    """
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", value, re.DOTALL)
    if match:
        value = match.group(1).strip()
    return replace_submission_name(value, _id), value




