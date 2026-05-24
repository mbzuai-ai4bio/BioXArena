PROMPT_TEMPLATE = """
You are a Machine Learning coding agent evaluated on a biomedical ML benchmark: BioXArena.

# Task & Paths
Task directory: {task_dir}  
Output directory: {output_dir}  
Description file: description.md

# Requirements
1. Explore and understand the public data first. Only use public files under {task_dir}.
2. Public files may include:
   - Training data: train.csv / train.jsonl.gz / train_*.npz / ...
   - Test data: test.csv / test.jsonl.gz / test_*.npz / ...
   - Modality-specific assets
   - sample_submission.csv
3. Perform appropriate feature engineering, preprocessing, and model selection for the task.
4. Train a concrete ML or deep learning model. 
   - If using deep learning, train on GPU(s). 4 GPUs are available.
5. Generate predictions on the test set.
6. Save the submission as `{output_dir}/submission.csv`.
   - Format must exactly match `sample_submission.csv`, including the exact number of columns, exact column names, and the exact values and order of the first column.
7. Save a metrics file as `{output_dir}/metrics.json` with this exact format:
{{
  "solution_generation_time_sec": "time from the beginning of code generation until the final correct solution.py is produced",
  "train_time_sec": "time spent on training/fitting the final correct solution.py",
  "test_time_sec": "time spent on running inference/prediction on the test set for the final correct solution.py",
  "code_total_time_sec": "solution_generation_time_sec + train_time_sec + test_time_sec",
  "input_tokens": "total number of input/prompt tokens used to complete the task across all API attempts",
  "output_tokens": "total number of output/completion tokens used to complete the task across all API attempts",
  "model_used": "e.g. XGBoost / RandomForest / CNN / Transformers / ...",
  "model_param_count": "number of trainable parameters if applicable; otherwise null or an estimated/appropriate value for the model",
  "notes": "any relevant details about feature engineering, preprocessing, etc."
}}
8. Save your final or best-performing Python solution to `{output_dir}/solution.py`, including the full training and inference code.
9. If applicable, save the corresponding trained model weights in `{output_dir}`. Store them as at least one model artifact file (for example `.pt`, `.pth`, `.ckpt`, `.bin`, `.joblib`, `.pkl`, `.pickle`, `.onnx`, `.model`, `.cbm`) or in a dedicated `weights/` or `checkpoints/` directory under `{output_dir}`.
10. Do NOT access any private directories, answer files, hidden labels, or non-public artifacts.
11. You are finished only after `submission.csv`, `metrics.json`, and `solution.py` are correctly saved in `{output_dir}`.

# Timing
Use Python's `time` module to record:
- train_time: total model training time
- test_time: total model testing time

The outer runner may overwrite `code_total_time_sec` in `metrics.json` to include end-to-end generation and execution time.
The outer runner may also inject `solution_generation_time_sec`.
The outer runner may also inject `input_tokens` and `output_tokens`.

# Execution Guidelines
- Show reasoning for preprocessing, model choice, and hyperparameters.
- Optimize model selection for the specific modality and task type (classification/regression/multi-label/survival/etc.)
- Use GPU(s) for deep learning tasks.
- If using PyTorch, explicitly move the model and all input tensors to CUDA and ensure training and inference run on GPU.
- Ensure output paths exist or create them.
- Handle various data formats (.csv, .jsonl.gz, .npz, etc.)
- Log any important notes in metrics.json.

# Task Description
Refer to the task-specific description below for details.

--- description.md ---
{description}
--- end description.md ---
"""
