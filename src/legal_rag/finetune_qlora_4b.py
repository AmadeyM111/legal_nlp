#!/usr/bin/env python3
import json
import argparse
import sys
from pathlib import Path
import re
import gzip
from typing import Any, Iterator
import time
import warnings
from contextlib import contextmanager

# GGUF — это формат весов llama.cpp (инференс), не совместимый с обучением MLX-LM/Transformers.
def _looks_like_gguf_ref(model_ref: str) -> bool:
    s = str(model_ref or "").strip().lower()
    if not s:
        return False
    if s.endswith(".gguf"):
        return True
    # Типичный нейминг HF-реп: *_gguf или */*_gguf
    return bool(re.search(r"(^|[\/_\-])gguf($|[\/_\-])", s))

# Нормализует разные форматы обучающих примеров (case/question/input/article/answer/output)
# в единую пару (user_prompt, assistant_answer) для чат‑формата дообучения.
def extract_user_and_answer(item: dict) -> tuple[str, str]:
    instruction = (item.get("instruction") or "").strip()
    input_text = (item.get("input") or "").strip()
    user_content = (
        (item.get("case") or "").strip()
        or (item.get("question") or "").strip()
        or input_text
        or (item.get("article_title") or "").strip()
    )
    if instruction and input_text:
        user_content = f"{instruction}\n\n{input_text}"
    elif instruction and not user_content:
        user_content = instruction

    assistant_content = (
        (item.get("article") or "").strip()
        or (item.get("answer") or "").strip()
        or (item.get("output") or "").strip()
        or (item.get("context") or "").strip()
    )
    return user_content, assistant_content

def _open_maybe_gzip_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")

def iter_training_records(path: str | Path) -> Iterator[dict]:
    """
    Поддерживаемые форматы:
    - *.jsonl / *.jsonl.gz: JSON Lines (по одному JSON-объекту на строку)
    - *.json: JSON-массив объектов (legacy)
    """
    p = Path(path)
    suffixes = p.suffixes
    is_jsonl = (suffixes[-1:] == [".jsonl"]) or (suffixes[-2:] == [".jsonl", ".gz"])

    if is_jsonl:
        with _open_maybe_gzip_text(p) as f:
            for line_no, line in enumerate(f, 1):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if line_no == 1:
                    s = s.lstrip("\ufeff")
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_no} in {p}") from e
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected JSON object on line {line_no} in {p}, got {type(obj).__name__}")
                yield obj
        return

    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Training data in {p} must be a JSON array (use JSONL for streaming)")
    for i, obj in enumerate(raw):
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object at index {i} in {p}, got {type(obj).__name__}")
        yield obj

def normalize_item_to_messages(item: dict) -> list[dict] | None:
    """
    Нормализует запись к chat-формату:
      [{"role": "...", "content": "..."}, ...]
    Поддерживает:
    - новый формат: {"messages": [...]}
    - legacy-форматы (через extract_user_and_answer)
    """
    messages = item.get("messages")
    if isinstance(messages, list):
        cleaned: list[dict] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "").strip()
            content = str(m.get("content") or "").strip()
            if not role or not content:
                continue
            cleaned.append({"role": role, "content": content})
        if not cleaned:
            return None
        if cleaned[-1]["role"] != "assistant":
            return None
        if not any(m["role"] == "user" for m in cleaned):
            return None
        return cleaned

    user_content, assistant_content = extract_user_and_answer(item)
    if not user_content.strip() or not assistant_content.strip():
        return None
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

def _messages_to_fallback_inst_text(messages: list[dict]) -> str:
    system_parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    system = "\n\n".join(system_parts).strip()
    last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user" and m.get("content")), "")
    last_assistant = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "assistant" and m.get("content")), ""
    )
    if system and last_user:
        last_user = f"{system}\n\n{last_user}"
    return f"<s>[INST] {last_user} [/INST] {last_assistant}</s>"

def messages_to_text(messages: list[dict], tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    return _messages_to_fallback_inst_text(messages)

# Автоопределяет доступный бэкенд обучения/инференса:
# - `mlx` на чипах Apple (M‑серия; самый быстрый вариант на Mac)
# - `cuda` на видеокартах NVIDIA
# - `mps` на графике Apple через PyTorch
# Возвращает строковый идентификатор, который дальше выбирает ветку импорта и обучения.
def install_backend():
    try:
        import mlx
        return "mlx"
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"  # запасной вариант (окружение без GPU)

BACKEND = install_backend()

if BACKEND == "mlx":
    # Чипы Apple (M‑серия) — самый быстрый вариант
    import types
    import mlx_lm.lora as mlx_lora
    from mlx_lm.tuner.datasets import ChatDataset
    print("Apple Silicon detected → MLX-LM")

    # Находит самый свежий файл весов LoRA-адаптера MLX-LM в `output_dir`:
    # - предпочитает файлы вида `{iter}_adapters.safetensors`
    # - иначе берёт `adapters.safetensors`, если он существует.
    def _find_latest_mlx_adapter_checkpoint(output_dir: Path) -> tuple[Path | None, int | None]:
        """
        Возвращает (path, iter) для самого свежего `*_adapters.safetensors` в `output_dir`.
        Если таких файлов нет, использует `adapters.safetensors`, если он существует (iter=None).
        """
        candidates: list[tuple[int, Path]] = []
        for p in output_dir.glob("*_adapters.safetensors"):
            m = re.search(r"(?P<it>\d+)_adapters\.safetensors$", p.name)
            if not m:
                continue
            try:
                candidates.append((int(m.group("it")), p))
            except ValueError:
                continue

        if candidates:
            it, p = max(candidates, key=lambda x: x[0])
            return p, it

        latest = output_dir / "adapters.safetensors"
        if latest.exists():
            return latest, None

        return None, None

    # Проверяет файл весов (safetensors) на NaN/Inf, чтобы не продолжать обучение
    # с "поломанными" весами (обычно это признак дивергенции обучения).
    def _first_nonfinite_in_safetensors(path: Path) -> tuple[str, str, int, int] | None:
        """
        Если в каком-либо тензоре есть не‑конечные значения, возвращает:
        (tensor_name, dtype, n_nans, n_infs).
        """
        try:
            from safetensors import safe_open
        except Exception:
            return None

        import numpy as np

        with safe_open(str(path), framework="numpy") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if np.issubdtype(t.dtype, np.floating) and (not np.isfinite(t).all()):
                    return (k, str(t.dtype), int(np.isnan(t).sum()), int(np.isinf(t).sum()))
        return None

    def _archive_mlx_checkpoints(output_dir: Path, resume_path: Path | None) -> Path | None:
        to_move: list[Path] = []
        for p in output_dir.glob("*_adapters.safetensors"):
            to_move.append(p)
        latest = output_dir / "adapters.safetensors"
        if latest.exists():
            to_move.append(latest)
        if not to_move:
            return resume_path

        ts = time.strftime("%Y%m%d_%H%M%S")
        archive_dir = output_dir / f"archive_{ts}"
        archive_dir.mkdir(parents=True, exist_ok=True)

        resume_resolved = resume_path.resolve() if resume_path else None
        new_resume_path = resume_path
        for p in to_move:
            p_resolved = p.resolve()
            target = archive_dir / p.name
            p.replace(target)
            if resume_resolved and p_resolved == resume_resolved:
                new_resume_path = target

        print(f"Archived {len(to_move)} existing adapter checkpoints to {archive_dir}")
        return new_resume_path

    @contextmanager
    def _suppress_tokenizer_length_warning():
        # HF tokenizers warn when len(ids) > tokenizer.model_max_length and truncation is disabled.
        # Here we may intentionally tokenize overlong sequences to estimate length / chunk them.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Token indices sequence length is longer than the specified maximum sequence length for this model.*",
                category=UserWarning,
            )
            yield

    def _estimate_token_count(messages: list[dict], tokenizer: Any) -> int:
        try:
            text = messages_to_text(messages, tokenizer)
        except Exception:
            text = " ".join(m.get("content", "") for m in messages)
        try:
            with _suppress_tokenizer_length_warning():
                return len(tokenizer.encode(text))
        except Exception:
            try:
                with _suppress_tokenizer_length_warning():
                    return len(tokenizer(text)["input_ids"])
            except Exception:
                return max(1, len(text.split()))

    def _tokenize_text(text: str, tokenizer: Any) -> list[int]:
        try:
            with _suppress_tokenizer_length_warning():
                return list(tokenizer.encode(text))
        except Exception:
            try:
                with _suppress_tokenizer_length_warning():
                    return list(tokenizer(text)["input_ids"])
            except Exception:
                return []

    def _decode_tokens(tokens: list[int], tokenizer: Any) -> str:
        try:
            return str(tokenizer.decode(tokens))
        except Exception:
            return ""

    def _chunk_overlong_example_preserve_all_tokens(
        messages: list[dict],
        tokenizer: Any,
        effective_max_seq_length: int,
        add_continuation_hint: bool = True,
    ) -> list[list[dict]]:
        """
        Разбивает слишком длинный пример на несколько, чтобы:
        - не терять ни одного токена из последнего assistant-сообщения
        - не превышать effective_max_seq_length (и не ловить Metal OOM)

        Реализация рассчитана на типичный формат [user..., assistant(last)].
        Для частей > 1 добавляем короткую подсказку в последнее user-сообщение,
        чтобы модель училась "продолжать" ответ.
        """
        if not messages or messages[-1].get("role") != "assistant":
            return [messages]
        assistant_text = str(messages[-1].get("content") or "")
        if not assistant_text.strip():
            return [messages]

        answer_tokens = _tokenize_text(assistant_text, tokenizer)
        if not answer_tokens:
            return [messages]

        last_user_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                last_user_idx = idx
                break

        out: list[list[dict]] = []
        part = 1
        start = 0
        while start < len(answer_tokens):
            # Собираем "промпт" для текущей части (assistant-контент пустой),
            # считаем доступное окно токенов под фрагмент ответа.
            prompt_messages = [dict(m) for m in messages]
            if add_continuation_hint and part > 1 and last_user_idx is not None:
                suffix = f"\n\nПродолжи ответ. Часть {part}."
                prompt_messages[last_user_idx] = {
                    "role": "user",
                    "content": str(prompt_messages[last_user_idx].get("content") or "") + suffix,
                }
            prompt_messages[-1] = {"role": "assistant", "content": ""}

            base_len = _estimate_token_count(prompt_messages, tokenizer)
            available = int(effective_max_seq_length) - int(base_len)
            if available < 64:
                # Если промпт сам по себе уже почти не помещается, chunking не поможет.
                return [messages]

            chunk_tokens = answer_tokens[start : start + available]
            chunk_text = _decode_tokens(chunk_tokens, tokenizer).strip()
            if not chunk_text:
                break

            prompt_messages[-1] = {"role": "assistant", "content": chunk_text}
            out.append(prompt_messages)
            part += 1
            start += available

        return out or [messages]

    # Запускает дообучение через MLX-LM LoRA:
    # - грузит базовую модель/токенизатор
    # - валидирует и нормализует JSON-датасет в формат `messages` (чат)
    # - делит данные на обучающую и валидационную части, настраивает LoRA и параметры обучения
    # - поддерживает возобновление с последнего *_adapters.safetensors (или заданного файла).
    def finetune(args):
        if _looks_like_gguf_ref(args.model):
            raise RuntimeError(
                "GGUF модели (llama.cpp) не поддерживаются в MLX-ветке обучения. "
                "Для fine-tune через MLX-LM нужен MLX-чекпойнт (веса в формате MLX), "
                "либо конвертация из обычной HF/Transformers модели в MLX (см. `python -m mlx_lm.convert -h`). "
                "Репозиторий вида '*_gguf' подходит только для инференса через llama.cpp."
            )

        print(f"Loading model: {args.model}")
        load_kwargs = {
            "tokenizer_config": {
                "trust_remote_code": True,
                "fix_mistral_regex": True,
            },
        }
        if args.mlx_load_in_8bit is not None:
            load_kwargs["load_in_8bit"] = bool(args.mlx_load_in_8bit)
        try:
            model, tokenizer = mlx_lora.load(args.model, **load_kwargs)
        except TypeError as e:
            if "load_in_8bit" in str(e):
                print("Warning: mlx_lm.lora.load does not support load_in_8bit; ignoring.")
                load_kwargs.pop("load_in_8bit", None)
                model, tokenizer = mlx_lora.load(args.model, **load_kwargs)
            else:
                raise
        model_max_len = getattr(tokenizer, "model_max_length", None)
        if isinstance(model_max_len, int) and model_max_len > 0 and args.max_seq_length > model_max_len:
            if bool(getattr(args, "_max_seq_length_was_passed", False)):
                print(
                    f"Warning: --max_seq_length={args.max_seq_length} exceeds model max length {model_max_len}. "
                    f"Clamping to {model_max_len}."
                )
            args.max_seq_length = model_max_len
        if getattr(args, "mlx_effective_max_seq_length", None) is not None:
            if isinstance(model_max_len, int) and model_max_len > 0 and args.mlx_effective_max_seq_length > model_max_len:
                if bool(getattr(args, "_mlx_effective_max_seq_length_was_passed", False)):
                    print(
                        f"Warning: --mlx_effective_max_seq_length={args.mlx_effective_max_seq_length} exceeds model max length "
                        f"{model_max_len}. Clamping to {model_max_len}."
                    )
                args.mlx_effective_max_seq_length = model_max_len

        mlx_effective_max_seq_length = int(getattr(args, "mlx_effective_max_seq_length", None) or args.max_seq_length)
        if mlx_effective_max_seq_length <= 0:
            raise ValueError("--mlx_effective_max_seq_length must be > 0")
        if mlx_effective_max_seq_length != args.max_seq_length:
            print(
                f"MLX effective max_seq_length: {mlx_effective_max_seq_length} "
                f"(requested --max_seq_length={args.max_seq_length})"
            )

        print(f"Loading training data: {args.data}")
        messages_data: list[dict] = []
        dropped_long = 0
        chunked = 0
        debug_raw_item: dict | None = None
        debug_original_messages: list[dict] | None = None
        debug_parts: list[list[dict]] | None = None
        for i, item in enumerate(iter_training_records(args.data)):
            messages = normalize_item_to_messages(item)
            if messages is None:
                print(f"Warning: Skipping item {i} with invalid/empty messages")
                continue

            # Быстрый пред-фильтр, чтобы не токенизировать каждый пример (это может заметно замедлить старт).
            # Если суммарный размер контента существенно меньше окна, он почти наверняка уместится по токенам.
            content_chars = 0
            for m in messages:
                content_chars += len(str(m.get("content") or ""))
            maybe_overlong = content_chars + 512 > mlx_effective_max_seq_length

            length = None
            if maybe_overlong and (args.mlx_drop_long_sequences or getattr(args, "mlx_chunk_long_sequences", False)):
                length = _estimate_token_count(messages, tokenizer)

            if args.mlx_drop_long_sequences:
                if length is None:
                    length = _estimate_token_count(messages, tokenizer)
                if length > mlx_effective_max_seq_length:
                    dropped_long += 1
                    continue

            if getattr(args, "mlx_chunk_long_sequences", False):
                if length is not None and length > mlx_effective_max_seq_length:
                    parts = _chunk_overlong_example_preserve_all_tokens(
                        messages,
                        tokenizer,
                        effective_max_seq_length=mlx_effective_max_seq_length,
                        add_continuation_hint=True,
                    )
                    if len(parts) > 1:
                        chunked += (len(parts) - 1)
                    for p in parts:
                        messages_data.append({"messages": p})
                    if getattr(args, "debug_show_example", False) and i == int(getattr(args, "debug_example_index", 0)):
                        debug_raw_item = item
                        debug_original_messages = messages
                        debug_parts = parts
                    continue

            messages_data.append({"messages": messages})
            if getattr(args, "debug_show_example", False) and i == int(getattr(args, "debug_example_index", 0)):
                debug_raw_item = item
                debug_original_messages = messages
                debug_parts = [messages]

        if len(messages_data) == 0:
            raise ValueError("No valid training data found after filtering")

        if dropped_long:
            print(
                f"Dropped {dropped_long} examples longer than {mlx_effective_max_seq_length} tokens "
                f"(--mlx_drop_long_sequences)."
            )
        if chunked:
            print(
                f"Chunked {chunked} overlong examples into additional parts to fit "
                f"{mlx_effective_max_seq_length} tokens (--mlx_chunk_long_sequences)."
            )

        if getattr(args, "debug_show_example", False):
            ex_idx = int(getattr(args, "debug_example_index", 0))
            max_chars = int(getattr(args, "debug_max_chars", 1200))
            show_token_ids = bool(getattr(args, "debug_show_token_ids", False))
            exit_after = bool(getattr(args, "debug_exit_after_example", True))

            if debug_original_messages is None or debug_parts is None:
                print(f"DEBUG: example index {ex_idx} was not found (dataset may be shorter or item was skipped).")
            else:
                print(f"\n=== DEBUG: processed example index {ex_idx} ===")
                if isinstance(debug_raw_item, dict):
                    print("DEBUG: raw keys:", ", ".join(sorted(map(str, debug_raw_item.keys()))))
                print(f"DEBUG: parts: {len(debug_parts)} (chunking={'on' if len(debug_parts) > 1 else 'off'})")
                for part_no, part_messages in enumerate(debug_parts, 1):
                    text = messages_to_text(part_messages, tokenizer)
                    token_ids = _tokenize_text(text, tokenizer)
                    print(f"\n--- Part {part_no}/{len(debug_parts)} ---")
                    print(f"tokens={len(token_ids)}; effective_max_seq_length={mlx_effective_max_seq_length}")
                    if len(token_ids) > mlx_effective_max_seq_length:
                        print("WARNING: tokenized length exceeds effective max_seq_length; this would be unsafe.")
                    if show_token_ids:
                        preview = token_ids[:80]
                        print("token_ids[:80] =", preview)
                    if max_chars > 0:
                        snippet = text[:max_chars]
                        if len(text) > max_chars:
                            snippet += "\n...<truncated>..."
                        print(snippet)
                print("\n=== DEBUG END ===\n")
            if exit_after:
                print("DEBUG: exiting before training (--debug_exit_after_example).")
                return

        print(f"Training with {len(messages_data)} examples")

        # MLX-LM (>=0.29) тренирует LoRA и сохраняет адаптеры, а не "слитую" модель.
        # Формат датасета: {"messages": [{"role": "...", "content": "..."}, ...]}
        if len(messages_data) < max(2, args.batch * 2):
            raise ValueError(
                f"Need at least {max(2, args.batch * 2)} examples for MLX training/validation split "
                f"(got {len(messages_data)}). Reduce --batch or add more data."
            )

        # Минимальная валидация, чтобы шаги eval не падали на пустом вал-сете.
        valid_size = max(args.batch, int(0.05 * len(messages_data)))
        valid_size = min(valid_size, len(messages_data) - args.batch)
        if valid_size < args.batch:
            raise ValueError(
                f"Not enough examples to create a validation split with batch_size={args.batch}. "
                "Reduce --batch or add more data."
            )
        valid_set = ChatDataset(messages_data[-valid_size:], tokenizer, mask_prompt=False)
        train_set = ChatDataset(messages_data[:-valid_size], tokenizer, mask_prompt=False)

        mlx_cfg = dict(mlx_lora.CONFIG_DEFAULTS)
        mlx_cfg.update(
            {
                "model": args.model,
                "train": True,
                "test": False,
                "fine_tune_type": "lora",
                "adapter_path": args.output,
                "batch_size": args.batch,
                "iters": args.iters,
                "learning_rate": args.lr,
                "max_seq_length": mlx_effective_max_seq_length,
                "steps_per_report": args.logging_steps,
                "steps_per_eval": max(args.save_steps, args.logging_steps),
                "val_batches": args.mlx_val_batches,
                "save_every": args.save_steps,
                "num_layers": args.mlx_num_layers,
                "grad_checkpoint": bool(args.mlx_grad_checkpoint),
                "grad_accumulation_steps": int(args.mlx_grad_accumulation_steps),
                "lora_parameters": {
                    "rank": args.rank,
                    "dropout": 0.05,
                    "scale": float(args.lora_alpha),
                },
            }
        )

        if args.mlx_wired_limit_gb and args.mlx_wired_limit_gb > 0:
            try:
                import mlx.core as mx

                mx.set_wired_limit(int(args.mlx_wired_limit_gb * (1024**3)))
                print(f"MLX wired limit override: {args.mlx_wired_limit_gb} GB")
            except Exception as e:
                print(f"Warning: failed to set MLX wired limit: {e}")

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        resume_path: Path | None = None
        resume_iter: int | None = None

        if args.mlx_resume_adapter_file:
            resume_path = Path(args.mlx_resume_adapter_file)
            if not resume_path.exists():
                raise FileNotFoundError(f"MLX resume adapter file not found: {resume_path}")
        elif args.resume_from_checkpoint:
            resume_path, resume_iter = _find_latest_mlx_adapter_checkpoint(output_dir)

        if resume_path is not None:
            if args.mlx_archive_existing_checkpoints:
                resume_path = _archive_mlx_checkpoints(output_dir, resume_path)
            if resume_iter is not None and args.iters > resume_iter:
                iters_to_run = args.iters - resume_iter
                print(
                    f"Resuming MLX LoRA from {resume_path} (checkpoint iter {resume_iter}); "
                    f"running remaining iters: {iters_to_run} (target total: {args.iters})"
                )
            else:
                iters_to_run = args.iters
                suffix = f" (checkpoint iter {resume_iter})" if resume_iter is not None else ""
                print(
                    f"Resuming MLX LoRA from {resume_path}{suffix}; "
                    f"running iters: {iters_to_run} (interpreted as additional)"
                )
            mlx_cfg["resume_adapter_file"] = str(resume_path)
            bad = _first_nonfinite_in_safetensors(resume_path)
            if bad is not None:
                name, dtype, n_nans, n_infs = bad
                raise ValueError(
                    "Refusing to resume: adapter checkpoint contains non-finite values "
                    f"(first: {name}, dtype={dtype}, nans={n_nans}, infs={n_infs}).\n"
                    "This usually means training diverged; start from scratch in a new --output "
                    "or reduce --lr / --lora_alpha, and save checkpoints more frequently early on."
                )
        else:
            iters_to_run = args.iters
            if (not args.overwrite_output_dir) and (not args.resume_from_checkpoint) and any(output_dir.iterdir()):
                raise ValueError(
                    f"Output dir is not empty: {output_dir}\n"
                    f"Use --resume_from_checkpoint to continue, or --overwrite_output_dir to start from scratch, "
                    f"or change --output."
                )

        mlx_cfg["iters"] = iters_to_run
        mlx_args = types.SimpleNamespace(**mlx_cfg)
        mlx_args.iters = iters_to_run
        if resume_path is not None:
            mlx_args.resume_adapter_file = str(resume_path)
            print("Note: MLX-LM iteration counter restarts at 1; weights are loaded, but step numbers are local.")
        mlx_lora.train_model(mlx_args, model, train_set, valid_set)
        try:
            tokenizer.save_pretrained(args.output)
        except Exception:
            pass
        print(f"Adapters saved to {args.output}")

else:
    # NVIDIA CUDA (а CPU — только как аварийный режим) — Transformers + PEFT + Unsloth (самый быстрый на CUDA)
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        AutoModelForCausalLM,
        DataCollatorForLanguageModeling,
    )
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, get_peft_model

    # Выбирает оптимизатор под текущее окружение:
    # - на CUDA пытается использовать 8‑битный AdamW из bitsandbytes (экономит видеопамять)
    # - иначе падает обратно на стандартный AdamW из PyTorch.
    def pick_optimizer() -> str:
        if not torch.cuda.is_available():
            return "adamw_torch"
        if not is_bitsandbytes_available():
            return "adamw_torch"
        try:
            from transformers.training_args import OptimizerNames

            valid = {o.value for o in OptimizerNames}
            for candidate in ("paged_adamw_8bit", "adamw_bnb_8bit", "adamw_8bit"):
                if candidate in valid:
                    return candidate
        except Exception:
            pass
        return "adamw_torch"

    # Проверяет доступность bitsandbytes (нужен для 4‑битной загрузки/обучения QLoRA).
    def is_bitsandbytes_available() -> bool:
        try:
            import bitsandbytes  # noqa: F401
            return True
        except Exception:
            return False
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported as _unsloth_is_bf16_supported
        USE_UNSLOTH = True
        print("Unsloth detected → до 2.5× быстрее + 70% меньше VRAM")
    except ImportError:
        USE_UNSLOTH = False
        print("Transformers + QLoRA")
        _unsloth_is_bf16_supported = None

    # Определяет, можно ли безопасно использовать bf16:
    # - если есть Unsloth, делегирует ему проверку
    # - иначе проверяет поддержку bf16 в текущей CUDA-сборке PyTorch.
    def is_bfloat16_supported() -> bool:
        if _unsloth_is_bf16_supported is not None:
            try:
                return bool(_unsloth_is_bf16_supported())
            except Exception:
                return False
        try:
            return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        except Exception:
            return False

    # Запускает дообучение через Transformers + PEFT (и Unsloth, если установлен):
    # - загружает модель в 4‑битном режиме (QLoRA), навешивает LoRA
    # - читает JSON-датасет, фильтрует "битые" примеры
    # - конвертирует пары (user/assistant) в строку формата [INST]...[/INST]
    # - токенизирует, настраивает Trainer, поддерживает возобновление по контрольным точкам.
    def finetune(args):
        if BACKEND == "mps":
            raise RuntimeError(
                "PyTorch MPS backend detected. This script's non-MLX path is configured for CUDA (QLoRA/4-bit). "
                "On Apple Silicon prefer installing MLX-LM and using BACKEND=mlx."
            )
        if BACKEND == "cpu":
            raise RuntimeError(
                "CPU-only environment detected. This script's non-MLX path is configured for CUDA (QLoRA/4-bit). "
                "Use an NVIDIA GPU (CUDA) or run the MLX backend on Apple Silicon."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA backend selected but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build/drivers or switch to MLX on Apple Silicon."
            )

        if "mlx" in str(args.model).lower():
            raise ValueError(
                "The selected model looks like an MLX-specific checkpoint, but CUDA/Transformers training was selected. "
                "Pass a regular Transformers/HF model id (e.g. mistralai/... or a local Transformers model directory)."
            )

        print(f"Loading model: {args.model}")
        if not is_bitsandbytes_available():
            raise RuntimeError(
                "bitsandbytes is required for 4-bit (QLoRA) loading/training. "
                "Install finetune dependencies or run MLX backend on Apple Silicon."
            )
        if USE_UNSLOTH:
            if args.gradient_checkpointing is None:
                use_gc = "unsloth"
            else:
                use_gc = "unsloth" if args.gradient_checkpointing else False
            model, tokenizer = FastLanguageModel.from_pretrained(
                args.model,
                dtype=None,  # авто bf16/fp16
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.rank,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                use_gradient_checkpointing=use_gc,
            )
        else:
            from peft import prepare_model_for_kbit_training
            from transformers import AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                fix_mistral_regex=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            )
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, LoraConfig(
                r=args.rank,
                lora_alpha=32,
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
            ))
            if args.gradient_checkpointing:
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Данные
        print(f"Loading training data: {args.data}")
        rows: list[dict] = []
        for i, item in enumerate(iter_training_records(args.data)):
            messages = normalize_item_to_messages(item)
            if messages is None:
                print(f"Warning: Skipping item {i} with invalid/empty messages")
                continue
            rows.append({"messages": messages})

        if len(rows) == 0:
            raise ValueError("No valid training data found after filtering")

        print(f"Processing {len(rows)} valid examples")
        
        dataset = Dataset.from_list(rows)

        # Превращает один пример датасета (словарь) в единое текстовое поле `text`,
        # которое Trainer затем будет токенизировать как обычный LM-датасет.
        def formatting_func(ex):
            return {"text": messages_to_text(ex["messages"], tokenizer)}
        dataset = dataset.map(formatting_func)

        # Токенизирует батч текстов (без паддинга), ограничивая длину `max_seq_length`.
        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=args.max_seq_length,
                padding=False,
            )

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
        )

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        last_checkpoint = None
        if args.resume_from_checkpoint:
            last_checkpoint = get_last_checkpoint(str(output_dir))
            if last_checkpoint is not None:
                print(f"Resuming from checkpoint: {last_checkpoint}")
            elif not args.overwrite_output_dir and any(output_dir.iterdir()):
                raise ValueError(
                    f"Output dir is not empty but no checkpoint found: {output_dir}\n"
                    f"Use --overwrite_output_dir to start from scratch or change --output."
                )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        gradient_accumulation_steps = max(1, 8 // max(1, args.batch))
        optim = pick_optimizer()

        trainer = Trainer(
            model=model,
            train_dataset=tokenized_dataset,
            args=TrainingArguments(
                per_device_train_batch_size=args.batch,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_train_epochs=args.epochs,
                learning_rate=args.lr,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=args.logging_steps,
                save_steps=args.save_steps,
                output_dir=args.output,
                overwrite_output_dir=args.overwrite_output_dir,
                optim=optim,
                gradient_checkpointing=bool(args.gradient_checkpointing),
                report_to="none",
                remove_unused_columns=False,
                save_total_limit=args.save_total_limit,
            ),
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        trainer.train(resume_from_checkpoint=last_checkpoint)
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print(f"Model saved to {args.output}")

# === Запуск ===
if __name__ == "__main__":
    default_model = (
        #"moot20/Mistral-Small-24B-Instruct-2501-MLX-8bits"
        #"IlyaGusev/saiga_llama3_8b"
        "Vikhrmodels/QVikhr-3-8B-Instruction-MLX_8bit"
        if BACKEND == "mlx"
        else "mistralai/Mistral-7B-Instruct-v0.3"
    )
    default_output = "models/QVikhr-3-8B-Instruction" if BACKEND == "mlx" else "models/mistral-7b-lora"
    #default_output = "models/Mistral-Small-24B" if BACKEND == "mlx" else "models/mistral-7b-lora"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/training/train.jsonl",
        help="Путь к train.jsonl (JSONL) или train.json (JSON-массив) с обучающими данными",
    )
    # parser.add_argument("--model", default="IlyaGusev/saiga_mistral_7b_merged", help="Базовая модель для дообучения")
    parser.add_argument(
        "--model",
        default=default_model,
        help="Базовая модель для дообучения (MLX: MLX-чекпойнт; CUDA: HF Transformers; GGUF не поддерживается).",
    )
    # parser.add_argument("--output", default="models/saiga-legal-7b", help="Директория вывода (модель/адаптеры)")
    parser.add_argument("--output", default=default_output, help="Директория вывода (модель/адаптеры)")
    parser.add_argument("--iters", type=int, default=2000, help="Количество итераций обучения (только MLX)")  # только для MLX
    parser.add_argument("--batch", type=int, default=4, help="Размер батча")
    parser.add_argument("--rank", type=int, default=64, help="Параметр LoRA rank")
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help="LoRA alpha (масштаб LoRA-обновления) для MLX; по умолчанию = 2*rank (только MLX)",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Скорость обучения (learning rate)")
    parser.add_argument("--epochs", type=int, default=3, help="Число эпох (только Transformers)")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Максимальная длина последовательности")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Включить gradient checkpointing (Transformers; для MLX используйте --mlx_grad_checkpoint)",
    )
    parser.add_argument(
        "--mlx_load_in_8bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Загружать MLX-модель в 8-битном виде, если поддерживается (только MLX; не GGUF).",
    )
    parser.add_argument(
        "--mlx_grad_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Использовать gradient checkpointing для экономии памяти (только MLX)",
    )
    parser.add_argument(
        "--mlx_grad_accumulation_steps",
        type=int,
        default=1,
        help="Шаги накопления градиента (только MLX)",
    )
    parser.add_argument(
        "--mlx_val_batches",
        type=int,
        default=1,
        help="Количество val-батчей на eval; -1 = весь val-set (только MLX)",
    )
    parser.add_argument(
        "--mlx_num_layers",
        type=int,
        default=-1,
        help="Сколько последних слоёв дообучать LoRA; -1 = все (только MLX)",
    )
    parser.add_argument(
        "--mlx_wired_limit_gb",
        type=float,
        default=0.0,
        help="Переопределить MLX Metal wired limit в GB (только MLX; 0 = не менять)",
    )
    parser.add_argument(
        "--mlx_resume_adapter_file",
        default=None,
        help="Путь к весам LoRA-адаптера (*.safetensors) для загрузки перед обучением (только MLX).",
    )
    parser.add_argument(
        "--mlx_archive_existing_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Архивировать существующие *_adapters.safetensors при resume, чтобы не перезаписывать их (только MLX).",
    )
    parser.add_argument(
        "--mlx_drop_long_sequences",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Пропускать примеры, превышающие эффективный max_seq_length для MLX (только MLX).",
    )
    parser.add_argument(
        "--mlx_chunk_long_sequences",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Разбивать слишком длинные примеры на части, чтобы не терять текст и не ловить OOM (только MLX).",
    )
    parser.add_argument(
        "--mlx_effective_max_seq_length",
        type=int,
        default=None,
        help="Фактический max_seq_length, который используется в MLX обучении (полезно против OOM); "
        "если не задан и --max_seq_length не передан, по умолчанию берётся 10240.",
    )
    parser.add_argument(
        "--debug_show_example",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Показать, как один пример превращается в текст/токены перед подачей в модель (только MLX).",
    )
    parser.add_argument(
        "--debug_example_index",
        type=int,
        default=0,
        help="Индекс примера в train.jsonl/json (0-based) для --debug_show_example (только MLX).",
    )
    parser.add_argument(
        "--debug_max_chars",
        type=int,
        default=1200,
        help="Сколько символов выводить из сериализованного текста (0 = не печатать текст).",
    )
    parser.add_argument(
        "--debug_show_token_ids",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Печатать первые 80 token ids для отладки (только MLX).",
    )
    parser.add_argument(
        "--debug_exit_after_example",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Выходить после показа примера, не начиная обучение (только MLX).",
    )
    parser.add_argument("--save_steps", type=int, default=200, help="Как часто сохранять checkpoint (в шагах)")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Макс. число checkpoint'ов для хранения")
    parser.add_argument("--logging_steps", type=int, default=20, help="Частота логирования (в шагах)")
    parser.add_argument(
        "--resume_from_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Продолжать с последнего checkpoint в --output (Transformers; для MLX берёт *_adapters.safetensors).",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Разрешить использовать непустой --output без resume (Transformers/MLX).",
    )
    args = parser.parse_args()

    # Проверяет, был ли флаг передан пользователем явно (в т.ч. как `--flag=value`).
    # Используется, чтобы выставлять "консервативные" дефолты под MLX только когда
    # пользователь не переопределил параметры сам.
    def _flag_was_passed(flag_name: str) -> bool:
        base = flag_name.lstrip("-")
        no_flag = f"--no-{base}"
        return any(
            a == flag_name or a.startswith(flag_name + "=") or a == no_flag or a.startswith(no_flag + "=")
            for a in sys.argv[1:]
        )

    args._max_seq_length_was_passed = _flag_was_passed("--max_seq_length")
    args._mlx_effective_max_seq_length_was_passed = _flag_was_passed("--mlx_effective_max_seq_length")

    # Консервативные значения по умолчанию для MLX на больших моделях, если пользователь не переопределил их явно.
    if BACKEND == "mlx":
        if not _flag_was_passed("--batch"):
            args.batch = 1
        if not _flag_was_passed("--max_seq_length"):
            args.max_seq_length = 12032
            if not _flag_was_passed("--mlx_effective_max_seq_length"):
                # На 8B+ моделях 12032 часто упирается в Metal OOM; 10240 обычно заметно стабильнее,
                # при этом сохраняет длинный контекст. Для ещё большей стабильности используйте 8192/6144.
                args.mlx_effective_max_seq_length = 10240
        if not _flag_was_passed("--rank"):
            args.rank = 16
        if not _flag_was_passed("--lora_alpha"):
            args.lora_alpha = float(args.rank * 2)
        if not _flag_was_passed("--lr"):
            args.lr = 2e-5
        if not _flag_was_passed("--save_steps"):
            args.save_steps = 200
        if not _flag_was_passed("--logging_steps"):
            args.logging_steps = 200

    # Проверяем, что входной файл с данными существует
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Training data file not found: {args.data}")

    if args.batch < 1:
        raise ValueError("--batch must be >= 1")
    if args.max_seq_length < 1:
        raise ValueError("--max_seq_length must be >= 1")
    if args.save_steps < 1:
        raise ValueError("--save_steps must be >= 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.iters < 1:
        raise ValueError("--iters must be >= 1")
    if BACKEND == "mlx":
        if args.lora_alpha is None:
            args.lora_alpha = float(args.rank * 2)
        if args.lora_alpha <= 0:
            raise ValueError("--lora_alpha must be > 0")
        if args.mlx_effective_max_seq_length is not None and args.mlx_effective_max_seq_length < 1:
            raise ValueError("--mlx_effective_max_seq_length must be >= 1")

    model_path = Path(args.model)
    if not model_path.exists() and ("/" not in args.model and "\\" not in args.model):
        candidate = Path("models") / args.model
        if candidate.exists():
            args.model = str(candidate)
            model_path = candidate

    if not model_path.exists():
        print(f"Note: model path not found locally, treating as HF model id: {args.model}")

    try:
        finetune(args)
        print(f"Готово → {args.output}")
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        raise
