# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#!/usr/bin/env python3
"""
Pre-compute PaliGemma anomaly scores for HIVAU training videos.

For each video in the training annotation, samples frames at 4 FPS,
creates 2×2 grids (4 frames per query), and runs PaliGemma batch inference
to produce per-query anomaly scores. These scores are saved as a JSON file
that the training DataLoader can load at fine-tuning time.

This ensures training-time PG context matches eval-time exactly.

Usage:
    python scripts/precompute/precompute_pg_scores.py --resume
    # Defaults to two workers on GPU pairs (0, 1) and (2, 3), batch 32 per pair.
    python scripts/precompute/precompute_pg_scores.py --resume --num-gpus 4 --gpus-per-worker 2
    python scripts/precompute/precompute_pg_scores.py --resume --num-gpus 1 --gpus-per-worker 1

Output:
    {output_path}/pg_scores_hivau_train.json
    Format: {
        "ucf-crime/clips/train/Abuse001_x264_E0C0.mp4": {
            "pg_scores": [0.7311, 0.6225, ...],
            "n_frames": 743,
            "sample_interval": 7,
            "num_queries": 27
        },
        ...
    }
"""

import os
import sys
import json
import multiprocessing as mp
import time
import signal
import logging
import warnings
import argparse
from pathlib import Path
import queue
import traceback
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm

import cv2
import PIL.Image
import torch
import torch.nn.functional as F

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# ===================== Grid Image Creation (identical to eval) =====================
GRID_SIZE = (2, 2)
USE_SEPARATOR = True
SEPARATOR_WIDTH = 2
SEPARATOR_COLOR = (128, 128, 128)


def compute_grid_params(image_size: int = 448):
    if USE_SEPARATOR:
        cell_size = (image_size - SEPARATOR_WIDTH) // 2
        total_size = cell_size * 2 + SEPARATOR_WIDTH
    else:
        cell_size = image_size // 2
        total_size = cell_size * 2
    return cell_size, total_size


def resize_frame(image, target_size):
    if image is None:
        return PIL.Image.new("RGB", (target_size, target_size), (0, 0, 0))
    return image.resize((target_size, target_size), PIL.Image.Resampling.LANCZOS)


def create_grid_image(frames, grid_size=(2, 2), image_size=448):
    if not frames:
        return PIL.Image.new("RGB", (image_size, image_size), (0, 0, 0))
    cell_size, total_size = compute_grid_params(image_size)
    rows, cols = grid_size
    expected_frames = rows * cols
    while len(frames) < expected_frames:
        frames.append(frames[-1] if frames else PIL.Image.new("RGB", (cell_size, cell_size)))
    if USE_SEPARATOR:
        grid_image = PIL.Image.new("RGB", (total_size, total_size), SEPARATOR_COLOR)
        for idx in range(expected_frames):
            row, col = divmod(idx, cols)
            resized = resize_frame(frames[idx], cell_size)
            x = col * (cell_size + SEPARATOR_WIDTH)
            y = row * (cell_size + SEPARATOR_WIDTH)
            grid_image.paste(resized, (x, y))
    else:
        grid_image = PIL.Image.new("RGB", (total_size, total_size))
        for idx in range(expected_frames):
            row, col = divmod(idx, cols)
            resized = resize_frame(frames[idx], cell_size)
            grid_image.paste(resized, (col * cell_size, row * cell_size))
    return grid_image


# ===================== Vision Encoder Utilities =====================
from safetensors.torch import load_file as safetensors_load_file
from peft import PeftModel
from accelerate import dispatch_model
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import types


def load_streamforest_vision_weights(model, weights_path):
    logging.info(f"Loading StreamForest vision weights from: {weights_path}")
    stream_weights = safetensors_load_file(weights_path)
    logging.info(f"  Loaded {len(stream_weights)} tensors from StreamForest")

    # Filter to vision_model keys only (match eval script)
    remapped = {}
    for key, value in stream_weights.items():
        new_key = key.replace("vision_tower.vision_tower.", "")
        if new_key.startswith("vision_model."):
            remapped[new_key] = value

    # Resize model's position embedding BEFORE loading weights (match eval script)
    pos_key = "vision_model.embeddings.position_embedding.weight"
    if pos_key in remapped:
        new_pos_shape = remapped[pos_key].shape
        old_pos_shape = model.vision_tower.vision_model.embeddings.position_embedding.weight.shape
        logging.info(f"  Position embedding: {old_pos_shape} -> {new_pos_shape}")

        new_num_positions = new_pos_shape[0]  # 729
        embed_dim = new_pos_shape[1]          # 1152

        model.vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(
            new_num_positions, embed_dim
        ).to(model.device).to(model.dtype)
        logging.info(f"  Resized position embedding to {new_num_positions} positions")

    # Load StreamForest weights
    missing, unexpected = model.vision_tower.load_state_dict(remapped, strict=False)
    logging.info(f"  Missing keys (kept from PaliGemma): {len(missing)}")
    logging.info(f"  Unexpected keys: {len(unexpected)}")

    model.vision_tower.vision_model.embeddings.num_positions = 729
    embeddings_module = model.vision_tower.vision_model.embeddings
    embeddings_module.register_buffer(
        "position_ids",
        torch.arange(729, device=model.device).expand((1, -1)),
        persistent=False,
    )
    model.config.vision_config.image_size = 384
    model.config.vision_config.num_image_tokens = 729
    model.config.vision_config.num_positions = 729
    model.config.text_config.num_image_tokens = 729
    logging.info(f"  Config updated: image_size=384, num_image_tokens=729")
    return model


def patch_vision_feature_layer(model, select_layer=-2):
    if hasattr(model, 'get_image_features'):
        target = model
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        target = model.base_model.model
    else:
        target = model

    def get_image_features_with_layer_select(self, pixel_values):
        vision_outputs = self.vision_tower(
            pixel_values, output_hidden_states=True)
        selected = vision_outputs.hidden_states[select_layer]
        image_features = self.multi_modal_projector(selected)
        image_features = image_features / (self.config.hidden_size ** 0.5)
        return image_features

    target.get_image_features = types.MethodType(
        get_image_features_with_layer_select, target)
    logging.info(f"  Patched get_image_features to use layer {select_layer}")
    return model


class PaliGemmaScorer:
    """Lightweight PaliGemma scorer for batch grid scoring."""

    def __init__(self, model_path, lora_path, device="cuda:0",
                 image_size=384, streamforest_weights_path=None,
                 vision_feature_layer=-2, attn_implementation="sdpa", gpu_ids=None):
        self.device = torch.device(device)
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else [self.device.index]
        self.torch_dtype = torch.bfloat16
        self.image_size = image_size
        self._logged_batch = False

        logging.info(f"[PaliGemma] Loading model from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        # Keep the scoring position at -1 even if padding is needed.
        self.processor.tokenizer.padding_side = "left"
        # Replace vision weights and merge the adapter before attaching dispatch hooks.
        # This also avoids loading a complete replica onto either GPU of a pair.
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=self.torch_dtype,
            device_map="cpu" if len(self.gpu_ids) > 1 else device,
            attn_implementation=attn_implementation)

        if image_size == 384 and streamforest_weights_path:
            self.model = load_streamforest_vision_weights(
                self.model, streamforest_weights_path)
            self.processor.image_processor.size = {"height": 384, "width": 384}
            self.processor.image_processor.image_seq_length = 729
            self.processor.image_seq_length = 729
            logging.info(f"[PaliGemma] Processor updated for 384x384")

        if lora_path and os.path.exists(lora_path):
            logging.info(f"[PaliGemma] Loading LoRA from {lora_path}...")
            self.model = PeftModel.from_pretrained(
                self.model, lora_path, torch_dtype=self.torch_dtype)
            self.model = self.model.merge_and_unload()
            logging.info("[PaliGemma] LoRA merged")

        if image_size == 384 and streamforest_weights_path and vision_feature_layer != -1:
            self.model = patch_vision_feature_layer(
                self.model, select_layer=vision_feature_layer)

        if len(self.gpu_ids) == 2:
            first, second = self.gpu_ids
            num_layers = len(self.model.language_model.model.layers)
            split = num_layers // 2
            device_map = {
                "vision_tower": first,
                "multi_modal_projector": first,
                "language_model.model.embed_tokens": first,
                "language_model.model.norm": second,
                # The output head shares its weights with the input embedding.
                "language_model.lm_head": first,
            }
            device_map.update({
                f"language_model.model.layers.{index}": first if index < split else second
                for index in range(num_layers)
            })
            self.model = dispatch_model(self.model, device_map=device_map,
                                        main_device=first)
            logging.info(
                "[PaliGemma] GPU pair %s: vision + embedding + layers 0-%d on GPU %d; "
                "layers %d-%d + norm on GPU %d; tied output head on GPU %d",
                self.gpu_ids, split - 1, first, split, num_layers - 1, second, first)

        self.model.eval()

        self.yes_token_id = self._get_token_id("Yes")
        self.no_token_id = self._get_token_id("No")
        self.yes_lower_token_id = self._get_token_id("yes")
        self.no_lower_token_id = self._get_token_id("no")
        logging.info(f"[PaliGemma] Ready. Yes={self.yes_token_id}, No={self.no_token_id}")

    def _get_token_id(self, word):
        tokens = self.processor.tokenizer.encode(word, add_special_tokens=False)
        return tokens[0] if tokens else None

    def batch_score_grids(self, grid_images, prompt, batch_size=32):
        """Score multiple grids in batches. Returns list of float scores."""
        if not grid_images:
            return []
        full_prompt = f"<image>{prompt}"
        all_scores = []
        for batch_start in range(0, len(grid_images), batch_size):
            batch_end = min(batch_start + batch_size, len(grid_images))
            batch_imgs = grid_images[batch_start:batch_end]
            real_batch_size = len(batch_imgs)
            # Keep every forward at the configured batch size, including tails.
            # BF16 GEMM kernels can otherwise vary with the tail's batch shape.
            # Duplicate images do not interact with real samples in evaluation.
            if real_batch_size < batch_size:
                batch_imgs = batch_imgs + [batch_imgs[-1]] * (batch_size - real_batch_size)
            prompts = [full_prompt] * len(batch_imgs)
            inputs = self.processor(
                text=prompts, images=batch_imgs,
                return_tensors="pt", padding=True).to(self.device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    num_logits_to_keep=1,
                    use_cache=False,
                )
                logits = outputs.logits
                if not self._logged_batch:
                    logging.info("[PaliGemma] GPUs %s: first inference batch=%d, logits=%s",
                                 self.gpu_ids, len(batch_imgs), tuple(logits.shape))
                    self._logged_batch = True
                for idx in range(real_batch_size):
                    last_logits = logits[idx, -1]
                    yes_logit = max(
                        last_logits[self.yes_token_id].float(),
                        last_logits[self.yes_lower_token_id].float())
                    no_logit = max(
                        last_logits[self.no_token_id].float(),
                        last_logits[self.no_lower_token_id].float())
                    probs = F.softmax(torch.stack([yes_logit, no_logit]), dim=0)
                    all_scores.append(round(probs[0].item(), 4))
            del inputs, outputs, logits
        return all_scores


def read_video_grids(video_path, image_size, target_fps=4, query_interval=4,
                     stop_event=None):
    """Read video frames and create grid images. CPU-bound, safe to run in thread."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_fps <= 0 or total_frames <= 0:
        cap.release()
        return None, None

    sample_interval = max(1, int(original_fps / target_fps))
    num_sampled_frames = (total_frames + sample_interval - 1) // sample_interval

    # Collect frame groups — sequential read with grab() to avoid costly H.264 seeking
    query_frame_groups = []
    current_group = []
    sampled_idx = 0
    current_frame_pos = 0

    while sampled_idx < num_sampled_frames:
        if stop_event is not None and stop_event.is_set():
            cap.release()
            return None, None
        target_frame = min(sampled_idx * sample_interval, total_frames - 1)

        # Skip forward sequentially (grab without decode)
        while current_frame_pos < target_frame:
            cap.grab()
            current_frame_pos += 1

        ret, frame = cap.read()
        current_frame_pos += 1

        if not ret:
            sampled_idx += 1
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = PIL.Image.fromarray(frame_rgb)
        current_group.append(pil_frame)

        if len(current_group) == query_interval or sampled_idx == num_sampled_frames - 1:
            query_frame_groups.append(current_group)
            current_group = []
        sampled_idx += 1
    cap.release()

    # Create grids
    grid_images = [
        create_grid_image(frames, GRID_SIZE, image_size)
        for frames in query_frame_groups
    ]

    metadata = {
        "n_frames": total_frames,
        "sample_interval": sample_interval,
    }
    del query_frame_groups
    return grid_images, metadata


def save_json_atomic(path, data):
    """Only the parent writes checkpoints; replace after a complete JSON write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w") as f:
        json.dump(data, f)
    os.replace(temporary_path, path)


def score_video_shard(rank, videos, args, pg_prompt, result_queue, stop_event):
    """Run one model replica per GPU group and report completed videos to parent."""
    # Spawn gives each worker a fresh CUDA context. The parent handles shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_event.set())
    signal.signal(signal.SIGUSR1, lambda signum, frame: stop_event.set())
    try:
        if stop_event.is_set():
            return
        gpu_ids = list(range(rank * args.gpus_per_worker,
                             (rank + 1) * args.gpus_per_worker))
        torch.cuda.set_device(gpu_ids[0])
        for gpu_id in gpu_ids:
            torch.cuda.reset_peak_memory_stats(gpu_id)
        # Each worker has its own reader pool; avoid OpenCV oversubscription.
        cv2.setNumThreads(1)
        logging.info("[Worker %d, GPUs %s] Loading model for %d videos",
                     rank, gpu_ids, len(videos))
        scorer = PaliGemmaScorer(
            model_path=args.paligemma_model_path,
            lora_path=args.paligemma_lora_path,
            device=f"cuda:{gpu_ids[0]}",
            gpu_ids=gpu_ids,
            image_size=args.paligemma_image_size,
            streamforest_weights_path=args.paligemma_streamforest_weights,
            vision_feature_layer=args.paligemma_vision_feature_layer,
            attn_implementation=args.paligemma_attn,
        )

        grid_buffer = []
        provenance = []

        def flush_buffer():
            if not grid_buffer:
                return
            scores = scorer.batch_score_grids(
                grid_buffer, pg_prompt, batch_size=args.batch_size)
            completed = {}
            for vpath, metadata, start, count in provenance:
                completed[vpath] = {
                    "pg_scores": scores[start:start + count],
                    "n_frames": metadata["n_frames"],
                    "sample_interval": metadata["sample_interval"],
                    "num_queries": count,
                }
            result_queue.put(("results", rank, completed))
            grid_buffer.clear()
            provenance.clear()

        def read_one(vpath):
            full_path = os.path.join(args.data_root, vpath)
            if not os.path.exists(full_path):
                return vpath, None, None, "file not found"
            try:
                grids, metadata = read_video_grids(
                    full_path, args.paligemma_image_size,
                    args.target_fps, args.query_interval, stop_event)
                error = "could not read video" if not grids else None
                return vpath, grids, metadata, error
            except Exception as exc:
                return vpath, None, None, str(exc)

        # Bounded CPU prefetch overlaps decoding with GPU scoring, across videos.
        with ThreadPoolExecutor(max_workers=args.num_readers) as pool:
            video_iter = iter(videos)
            pending = set()
            exhausted = False
            while not stop_event.is_set():
                while not exhausted and len(pending) < args.num_readers * 3:
                    vpath = next(video_iter, None)
                    if vpath is None:
                        exhausted = True
                        break
                    pending.add(pool.submit(read_one, vpath))
                if not pending:
                    break
                done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    if stop_event.is_set():
                        break
                    vpath, grids, metadata, error = future.result()
                    if error:
                        result_queue.put(("failed", rank, (vpath, error)))
                        continue
                    start = len(grid_buffer)
                    grid_buffer.extend(grids)
                    provenance.append((vpath, metadata, start, len(grids)))
                    if len(grid_buffer) >= args.batch_size:
                        flush_buffer()
                if exhausted and not pending:
                    break
            for future in pending:
                future.cancel()
        # Finish the accumulated videos before acknowledging a graceful stop.
        flush_buffer()
        for gpu_id in gpu_ids:
            logging.info("[Worker %d] GPU %d peak allocated=%.2f GiB, reserved=%.2f GiB",
                         rank, gpu_id,
                         torch.cuda.max_memory_allocated(gpu_id) / 2**30,
                         torch.cuda.max_memory_reserved(gpu_id) / 2**30)
    except BaseException:
        stop_event.set()
        result_queue.put(("error", rank, traceback.format_exc()))
        raise
    finally:
        result_queue.put(("done", rank, None))


def run_gpu_workers(videos, args, pg_prompt, results):
    """Merge worker results centrally so --resume keeps its existing JSON format."""
    context = mp.get_context("spawn")
    stop_event = context.Event()
    result_queue = context.Queue(maxsize=128)
    workers = []
    done_workers = set()
    failed = []
    errors = []
    processed_count = 0
    next_save = 500
    start_time = time.time()
    previous_handlers = {}

    def handle_shutdown(signum, frame):
        logging.warning("Received signal %d; saving after in-flight batches finish", signum)
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, handle_shutdown)

    pbar = tqdm(total=len(videos), desc="Processing videos (all GPUs)")
    try:
        # Disjoint shards ensure that no video is scored by two GPU groups.
        num_workers = args.num_gpus // args.gpus_per_worker
        for rank in range(min(num_workers, len(videos))):
            worker = context.Process(
                target=score_video_shard,
                args=(rank, videos[rank::num_workers], args, pg_prompt,
                      result_queue, stop_event),
                name=f"pg-worker-{rank}",
            )
            worker.start()
            workers.append(worker)

        while len(done_workers) < len(workers):
            try:
                kind, rank, payload = result_queue.get(timeout=1)
            except queue.Empty:
                # Detect crashes such as OOM kills that cannot send an error message.
                for rank, worker in enumerate(workers):
                    if rank not in done_workers and worker.exitcode is not None:
                        errors.append(f"Worker {rank} exited without completion (exit {worker.exitcode})")
                        done_workers.add(rank)
                        stop_event.set()
                continue

            if kind == "results":
                results.update(payload)
                count = len(payload)
            elif kind == "failed":
                failed.append(payload)
                logging.warning("[Worker %d] %s: %s", rank, *payload)
                count = 1
            elif kind == "error":
                errors.append(f"Worker {rank}:\n{payload}")
                logging.error("Worker %d failed:\n%s", rank, payload)
                stop_event.set()
                continue
            elif kind == "done":
                done_workers.add(rank)
                continue
            else:
                raise RuntimeError(f"Unknown GPU worker message: {kind}")

            processed_count += count
            pbar.update(count)
            if processed_count >= next_save:
                save_json_atomic(args.output_path, results)
                next_save = (processed_count // 500 + 1) * 500
                elapsed = time.time() - start_time
                rate = processed_count / max(elapsed, 1e-6)
                logging.info(
                    "Progress: %d/%d (%.1f min elapsed, ~%.1f min remaining)",
                    processed_count, len(videos), elapsed / 60,
                    (len(videos) - processed_count) / rate / 60)
    finally:
        stop_event.set()
        pbar.close()
        # Drain results during shutdown, including a partially started worker pool.
        while True:
            try:
                kind, rank, payload = result_queue.get(timeout=1)
            except queue.Empty:
                if not any(worker.is_alive() for worker in workers):
                    break
                continue
            if kind == "results":
                results.update(payload)
                processed_count += len(payload)
            elif kind == "failed":
                failed.append(payload)
                processed_count += 1
            elif kind == "error":
                errors.append(f"Worker {rank}:\n{payload}")
        for rank, worker in enumerate(workers):
            worker.join()
            if worker.exitcode:
                errors.append(f"Worker {rank} exited with code {worker.exitcode}")
        result_queue.close()
        result_queue.join_thread()
        save_json_atomic(args.output_path, results)
        output_path = Path(args.output_path)
        failed_path = output_path.with_name(output_path.stem + "_failed.json")
        if failed:
            save_json_atomic(failed_path, failed)
        elif failed_path.exists():
            failed_path.unlink()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    logging.info(
        "Pre-computation %s: %d total saved, %d/%d attempted this run, %d failed, %.1f min. Output: %s",
        "interrupted" if processed_count < len(videos) else "complete",
        len(results), processed_count, len(videos), len(failed),
        (time.time() - start_time) / 60, args.output_path)
    if failed:
        logging.info("Failed list: %s", failed_path)
    if errors:
        raise RuntimeError("GPU pre-computation failed; completed scores were saved.\n" + "\n".join(errors))


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute PaliGemma anomaly scores for HIVAU training videos")

    # PaliGemma configuration (must match eval exactly)
    default_project_root = os.environ.get(
        "REACTVAU_ROOT",
        str(Path(__file__).resolve().parents[2])
    )
    default_ckpt_root = os.environ.get(
        "CKPT_ROOT", os.path.join(default_project_root, "ckpt"))
    default_hivau_video_root = os.environ.get(
        "HIVAU_VIDEO_ROOT", "/path/to/HIVAU-70k/videos")

    parser.add_argument("--paligemma-model-path", type=str,
                        default=os.environ.get(
                            "PALIGEMMA_MODEL_PATH",
                            os.path.join(default_ckpt_root, "paligemma2-3b-mix-448")))
    parser.add_argument("--paligemma-lora-path", type=str,
                        default=os.environ.get(
                            "PALIGEMMA_LORA_PATH",
                            os.path.join(default_ckpt_root, "paligemma2-3b-vad-lora-384-combined-binary/training_384")))
    parser.add_argument("--paligemma-image-size", type=int, default=384)
    parser.add_argument("--paligemma-streamforest-weights", type=str,
                        default=os.environ.get(
                            "PALIGEMMA_SF_WEIGHTS",
                            os.path.join(default_ckpt_root, "extracted_weights/streamforest_vision_encoder_with_prefix.safetensors")))
    parser.add_argument("--paligemma-vision-feature-layer", type=int, default=-2)
    parser.add_argument("--paligemma-attn", type=str, default="sdpa")

    # Data configuration
    parser.add_argument("--anno-path", type=str,
                        default=os.environ.get("HIVAU_TRAIN_JSON", ""),
                        help="Downloaded HIVAU training annotation JSON path")
    parser.add_argument("--data-root", type=str,
                        default=default_hivau_video_root)

    # Pipeline configuration (must match eval exactly)
    parser.add_argument("--target-fps", type=int, default=4)
    parser.add_argument("--query-interval", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Inference batch size per model replica / GPU group (default: 32)")
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="Number of visible CUDA GPUs to use (default: 4)")
    parser.add_argument("--gpus-per-worker", type=int, choices=(1, 2), default=2,
                        help="GPUs per model replica; 4 GPUs / 2 gives groups (0,1), (2,3)")
    parser.add_argument("--num-readers", type=int, default=4,
                        help="CPU video reader threads per GPU group (default: 4)")

    # Output
    parser.add_argument("--output-path", type=str,
                        default=os.environ.get(
                            "PG_SCORES_PATH",
                            os.path.join(default_project_root, "precomputed/pg_scores_hivau_train.json")))

    # Resume support
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file (skip already processed videos)")

    args = parser.parse_args()

    for name in ("num_gpus", "num_readers", "batch_size", "query_interval"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.target_fps <= 0:
        parser.error("--target-fps must be positive")
    if args.num_gpus % args.gpus_per_worker:
        parser.error("--num-gpus must be divisible by --gpus-per-worker")
    if not args.anno_path:
        parser.error(
            "--anno-path is required; set HIVAU_TRAIN_JSON in paths.local.sh "
            "or pass --anno-path explicitly.")
    if not os.path.isfile(args.anno_path):
        parser.error(f"HIVAU training annotation not found: {args.anno_path}")

    # Setup
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # Add project root to path for get_prompt import
    sys.path.insert(0, default_project_root)
    from vad.get_prompt import get_grid_prompt

    # PG prompt (must match eval)
    pg_prompt = get_grid_prompt(add_special_tokens=False)
    logging.info(f"PG prompt: '{pg_prompt}'")

    # Load annotation
    logging.info(f"Loading annotation from {args.anno_path}")
    with open(args.anno_path) as f:
        annotations = json.load(f)

    # Extract unique video paths
    unique_videos = sorted(set(d["video"] for d in annotations if "video" in d))
    logging.info(f"Total annotation entries: {len(annotations)}")
    logging.info(f"Unique videos: {len(unique_videos)}")

    # Resume support
    existing_results = {}
    if args.resume and os.path.exists(args.output_path):
        with open(args.output_path) as f:
            existing_results = json.load(f)
        logging.info(f"Resuming: {len(existing_results)} videos already processed")
        unique_videos = [v for v in unique_videos if v not in existing_results]
        logging.info(f"Remaining: {len(unique_videos)} videos to process")

    if not unique_videos:
        logging.info("No videos remain; skipping GPU initialization")
        return

    visible_gpus = torch.cuda.device_count()
    if visible_gpus < args.num_gpus:
        parser.error(
            f"Requested {args.num_gpus} GPUs, but only {visible_gpus} CUDA GPUs are visible. "
            "Set CUDA_VISIBLE_DEVICES or lower --num-gpus.")
    logging.info(
        "Using %d GPUs in %d groups of %d, batch size %d per group, %d reader threads per group",
        args.num_gpus, args.num_gpus // args.gpus_per_worker, args.gpus_per_worker,
        args.batch_size, args.num_readers)
    run_gpu_workers(unique_videos, args, pg_prompt, existing_results)


if __name__ == "__main__":
    main()
