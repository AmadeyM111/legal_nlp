#!/usr/bin/env python3
import json
import argparse
import sys
from pathlib import Path
import re
from datasets import Dataset

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

    # Запускает дообучение через MLX-LM LoRA:
    # - грузит базовую модель/токенизатор
    # - валидирует и нормализует JSON-датасет в формат `messages` (чат)
    # - делит данные на обучающую и валидационную части, настраивает LoRA и параметры обучения
    # - поддерживает возобновление с последнего *_adapters.safetensors (или заданного файла).
    def finetune(args):
        print(f"Loading model: {args.model}")
        model, tokenizer = mlx_lora.load(
            args.model,
            tokenizer_config={
                "trust_remote_code": True,
                "fix_mistral_regex": True,
            },
        )

        print(f"Loading training data: {args.data}")
        with open(args.data, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            
        # Проверяем формат обучающих данных
        if not isinstance(raw, list):
            raise ValueError("Training data must be a JSON array")
        
        train_data = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                print(f"Warning: Skipping non-dict item at index {i}")
                continue
                
            user_content, assistant_content = extract_user_and_answer(item)
            
            if not user_content.strip() or not assistant_content.strip():
                print(f"Warning: Skipping item {i} with empty content")
                continue
                
            train_data.append([
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ])
        
        if len(train_data) == 0:
            raise ValueError("No valid training data found after filtering")
            
        print(f"Training with {len(train_data)} examples")

        # MLX-LM (>=0.29) тренирует LoRA и сохраняет адаптеры, а не "слитую" модель.
        # Формат датасета: {"messages": [{"role": "...", "content": "..."}, ...]}
        messages_data = [{"messages": messages} for messages in train_data]
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
                "max_seq_length": args.max_seq_length,
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
                    "scale": 32.0,
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
                    "or reduce --lr / LoRA scale, and save checkpoints more frequently early on."
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
                use_gradient_checkpointing="unsloth",
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

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Данные
        print(f"Loading training data: {args.data}")
        with open(args.data, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            
        # Проверяем формат обучающих данных
        if not isinstance(raw, list):
            raise ValueError("Training data must be a JSON array")
            
        # Отфильтровываем некорректные/пустые записи
        filtered_raw = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                print(f"Warning: Skipping non-dict item at index {i}")
                continue
                
            user_content, ans_content = extract_user_and_answer(item)
            
            if not user_content.strip() or not ans_content.strip():
                print(f"Warning: Skipping item {i} with empty content")
                continue
                
            filtered_raw.append(item)
            
        if len(filtered_raw) == 0:
            raise ValueError("No valid training data found after filtering")

        print(f"Processing {len(filtered_raw)} valid examples")
        
        dataset = Dataset.from_list(filtered_raw)

        # Превращает один пример датасета (словарь) в единое текстовое поле `text`,
        # которое Trainer затем будет токенизировать как обычный LM-датасет.
        def formatting_func(ex):
            user, ans = extract_user_and_answer(ex)
            text = f"<s>[INST] {user} [/INST] {ans}</s>"
            return {"text": text}
        dataset = dataset.map(lambda x: {"text": formatting_func(x)["text"]})

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
        "moot20/Mistral-Small-24B-Instruct-2501-MLX-8bits"
        if BACKEND == "mlx"
        else "mistralai/Mistral-7B-Instruct-v0.3"
    )
    default_output = "models/Mistral-Small-24B" if BACKEND == "mlx" else "models/mistral-7b-lora"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/training/train.json", help="Путь к JSON-файлу с обучающими данными")
    # parser.add_argument("--model", default="IlyaGusev/saiga_mistral_7b_merged", help="Базовая модель для дообучения")
    parser.add_argument("--model", default=default_model, help="Базовая модель для дообучения")
    # parser.add_argument("--output", default="models/saiga-legal-7b", help="Директория вывода (модель/адаптеры)")
    parser.add_argument("--output", default=default_output, help="Директория вывода (модель/адаптеры)")
    parser.add_argument("--iters", type=int, default=2000, help="Количество итераций обучения (только MLX)")  # только для MLX
    parser.add_argument("--batch", type=int, default=4, help="Размер батча")
    parser.add_argument("--rank", type=int, default=64, help="Параметр LoRA rank")
    parser.add_argument("--lr", type=float, default=2e-4, help="Скорость обучения (learning rate)")
    parser.add_argument("--epochs", type=int, default=3, help="Число эпох (только Transformers)")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Максимальная длина последовательности")
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
    parser.add_argument("--save_steps", type=int, default=200, help="Как часто сохранять checkpoint (в шагах)")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Макс. число checkpoint'ов для хранения")
    parser.add_argument("--logging_steps", type=int, default=10, help="Частота логирования (в шагах)")
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
        return any(a == flag_name or a.startswith(flag_name + "=") for a in sys.argv[1:])

    # Консервативные значения по умолчанию для MLX на больших моделях, если пользователь не переопределил их явно.
    if BACKEND == "mlx":
        if not _flag_was_passed("--batch"):
            args.batch = 1
        if not _flag_was_passed("--max_seq_length"):
            args.max_seq_length = 1024
        if not _flag_was_passed("--rank"):
            args.rank = 16
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
