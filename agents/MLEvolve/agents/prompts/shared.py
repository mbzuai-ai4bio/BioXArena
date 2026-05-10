"""Static shared prompt fragments."""

ROBUSTNESS_GENERALIZATION_STRATEGY = {
    "💡 Recommendation: Robustness & Generalization Strategy": [
        "",
        "**To improve model robustness and generalization on unseen data:**",
        "",
        "✅ **Architecture**: Match model inductive bias to data structure (e.g., CNNs/ViTs for spatial grids, Transformers/RNNs for sequences, GNNs/GCNs for graphs/topology)",
        "✅ **Input Strategy**: Handle variable-length or large-scale inputs via **windowing strategies** or patch-based processing (consider overlap for smoother predictions)",
        "✅ **Regularization**: Consider using Dropout, Batch/Layer Norm, Weight Decay, or Label Smoothing",
        "✅ **Loss Function**: Inspect class distribution and adapt loss accordingly (e.g., weighted loss, FocalLoss, or task-specific objectives)",
        "✅ **Learning Rate**: Consider using adaptive schedules like Cosine Annealing or ReduceLROnPlateau or Warmup with differential rates if needed",
        "✅ **Data Augmentation**: Apply domain-appropriate augmentation based on data modality (e.g., geometric transforms, masking, mixup)",
        "✅ **Validation**: Monitor validation metrics strictly and use early stopping to prevent overfitting",
        "",
        "⚠️ **Note**:",
        "Prioritize capturing the intrinsic structure of the data (Inductive Bias) over simply increasing model size.",
        "",
    ]
}


def prompt_leakage_prevention():
    """Data leakage prevention."""
    return {
        "🚨 DATA LEAKAGE PREVENTION": [
            "",
            "⚠️ **Strict Isolation Principle**: Validation/Test data must remain strictly unseen during training.",
            "",
            "✅ **Sequence**: Always **Split Data FIRST**, then apply processing.",
            "✅ **Stateful Transformations**: Fit all Scalers, Encoders, Imputers, and Tokenizers **ONLY on Training data**, then use `.transform()` on Validation/Test.",
            "✅ **Feature Engineering**: Calculate global statistics (e.g., mean, variance, vocabulary) solely from the Training set.",
            "✅ **Target Leakage**: Never use target information (e.g., Target Encoding) from the validation set.",
            "",
        ]
    }


def prompt_resp_fmt():
    """Response format for plan + code"""
    return {
        "Response format": (
            "Your response should be a brief outline/sketch of your proposed solution in natural language, "
            "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
            "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
        )
    }


def get_internet_clarification(pretrain_model_dir: str = ""):
    """Internet access clarification for improve/debug stages."""
    lines = [
        "**⚠️ IMPORTANT: Internet Access During Code Development**",
        "- The \"no internet access\" restriction mentioned in the task description applies **ONLY to submission evaluation after code generation** (for mle-bench test set).",
        "- **During code development, you CAN and SHOULD use online resources** such as torch.hub.load(), HuggingFace transformers, timm, etc.",
    ]
    if pretrain_model_dir:
        lines.append(
            f"- **Model paths under `{pretrain_model_dir}/` are GUARANTEED to exist and be available** (e.g., DINOv3, Siglip2 etc.). You can directly use them without `Path question`."
        )
    lines.append(
        "- **Do NOT question internet access concerns - all standard ML libraries and pretrained models are available during development."
    )
    return lines
