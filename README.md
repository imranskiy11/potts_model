# Potts Model — Rust + CUDA / CPU

| режим | как собрать | примечания |
|-------|-------------|-----------|
| **GPU / CUDA** (по-умолчанию) | `cargo run --release` | Требует CUDA 12.x, GPU ≥ sm_61 |
| **CPU-fallback**  | `cargo run --release --no-default-features --features cpu` | Однопоточная медленная проверка |
| **Обе реализации** (для бенчмарков) | `cargo run --release --features "cuda cpu"` | Собирает оба пути |

Параметры модели задаются в UI.  
GPU-ядро автоматически выбирает `sm_61`, если не передать `--gpu-arch`.
