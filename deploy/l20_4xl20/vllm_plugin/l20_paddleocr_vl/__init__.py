"""vLLM plugin entry point; does nothing unless L20_PADDLEOCR_VL_BATCHED=1."""
import os

ENV = "L20_PADDLEOCR_VL_BATCHED"


def register():
    if os.environ.get(ENV) != "1":
        return
    from vllm import ModelRegistry

    # Overrides the built-in architecture of the same name for this process only.
    ModelRegistry.register_model(
        "PaddleOCRVLForConditionalGeneration",
        "l20_paddleocr_vl.model:BatchedPaddleOCRVLForConditionalGeneration",
    )
