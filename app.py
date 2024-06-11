# FastAPI | Mixtral | PagedAttention using vLLM
import os
import time

import modal

MODEL_DIR = "/model"
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_REVISION = "1e637f2d7cb0a9d6fb1922f305cb784995190a83" # HF commit PR #150
GPU_CONFIG = modal.gpu.A100(size="80GB", count=2)

# Define a Modal container image | Download the weights to the Modal image
def download_model_to_image(model_dir, model_name, model_revision):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        revision=model_revision,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"] # Mixtral ~100GB with safetensors
    )
    move_cache()

# vLLM Dockerhub image with model weights
vllm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.4.0.post1",
        "torch==2.1.2",
        "transformers==4.39.3",
        "ray==2.10.0",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={
            "model_dir": MODEL_DIR,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
        },
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)

app = modal.App(
    "mixtral-vllm-fastapi"
)

# Define Model using vLLM
# https://modal.com/docs/guide/lifecycle-functions
@app.cls(
    gpu=GPU_CONFIG,
    timeout=60 * 10,  # timeout after 10 minutes
    allow_concurrent_inputs=10,
    image=vllm_image
)
class Model:
    @modal.enter()
    def start_engine(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        print("🥶 cold starting inference.  This will take a few minutes.")
        start = time.monotonic_ns()

        engine_args = AsyncEngineArgs(
            model=MODEL_DIR,
            tensor_parallel_size=GPU_CONFIG.count,
            gpu_memory_utilization=0.90,
            enforce_eager=False,  # capture graph for faster inference
            disable_log_stats=True,  # disable logging to stream tokens
            disable_log_requests=True,
        )
        self.template = "[INST] {user} [/INST]"

        # NOTE: Cold start loading the model will take a few minutes.
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        duration_s = (time.monotonic_ns() - start) / 1e9  # convert nanoseconds to seconds
        print(f"🚀 engine started in {duration_s:.0f}s")

    @modal.method()
    async def completion_stream(self, user_question):
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(
            temperature=0.75,
            max_tokens=128,  # generated output max tokens
            repetition_penalty=1.1,
        )

        request_id = random_uuid()
        result_generator = self.engine.generate(
            self.template.format(user=user_question),
            sampling_params,
            request_id,
        )

        index, num_tokens = 0, 0
        start = time.monotonic_ns()
        async for output in result_generator:
            if (
                output.outputs[0].text
                and "\ufffd" == output.outputs[0].text[-1]
            ):
                continue

            # get results from inference
            text_delta = output.outputs[0].text[index:]
            index = len(output.outputs[0].text)
            num_tokens = len(output.outputs[0].token_ids)

            yield text_delta

        duration_s = (time.monotonic_ns() - start) / 1e9

        yield (
            f"\n\tGenerated {num_tokens} tokens from {MODEL_NAME} in {duration_s:.1f}s,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.\n"
        )

    @modal.exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()

# Run Mixtral-8x7B
# NOTE: TBD: Cannot run locally on a Mac.  Requires CUDA (NVIDIA gpus).
@app.local_entrypoint()
def main():
    start = time.monotonic_ns()
    print(f"Launching app...")
    questions = [
        "Implement a Python function to compute Fibonacci numbers.",
        "What is the fable in involving a fox and grapes?",
        "What is the product of 9 and 8?",
        "Who is the current President of the United States?  Where is he from?",
        "Who is the current Vice President of the United States?",
        "Who is the current President and Primer of France?  Where is he from?",
        "Who is the current Prime Minister of Canada?  Where is he from?",
        "What is the capital of Texas?  Provide 3 fun adult things to do there.",
        "Have any United States presidents every been convicted on a crime?",
    ]
    model = Model()
    for question in questions:
        print("Sending new request:", question, "\n\n")
        for text in model.completion_stream.remote_gen(question):
            print(text, end="", flush=text.endswith("\n"))

    duration_s = (time.monotonic_ns() - start) / 1e9
    print(f"Completed text generation inference...")
    print(f"Time: {duration_s:.0f}seconds")
