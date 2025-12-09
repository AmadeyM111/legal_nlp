import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODEL_DIR = PROJECT_ROOT / "models" / "saiga_mistral_7b_merged"
MODEL_ID = "IlyaGusev/saiga_mistral_7b_merged"

def file_sha256(path: str, buf_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()

for name in ["pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin", "pytorch_model.bin.index.json"]:
    p = MODEL_DIR / name
    print(name, file_sha256(p))