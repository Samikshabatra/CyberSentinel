# Fine-tuning

## 1. What the fine-tuned model is responsible for

The model performs cybersecurity **reasoning**: classifying an event, extracting
the evidence that supports the classification, estimating severity and
confidence, and proposing candidate techniques.

It is **not** responsible for threat-intelligence facts. Those come from
retrieval (`docs/rag.md`). This separation is the central design decision of the
project:

| | Fine-tuning | RAG |
|---|---|---|
| Teaches | behaviour, format, domain reasoning | current, citable facts |
| Updated by | retraining | re-ingesting documents |
| Fails as | outdated knowledge baked into weights | missing or irrelevant retrieval |
| Used for | "is this brute force, and why?" | "what is T1110 and where is that documented?" |

Putting the ATT&CK corpus into the weights would make every knowledge update a
retraining job and would make citation impossible.

## 2. Why QLoRA

Full fine-tuning of a 3B parameter model needs far more GPU memory than a
consumer card provides. QLoRA makes it feasible:

1. Load the base model quantised to 4-bit (NF4) - roughly a quarter of the
   fp16 footprint.
2. Freeze every base weight.
3. Attach small trainable low-rank adapters to the attention and MLP
   projections.
4. Train only the adapters.

With rank 16 across seven projection modules, the trainable parameter count is
well under 1% of the model. The saved artefact is the adapter only - tens of
megabytes, not gigabytes - so experiments are cheap to store and compare, and
the application can switch between base and fine-tuned behaviour by changing one
setting.

```
base model ──► 4-bit quantisation ──► freeze base ──► attach LoRA adapters
                                                            │
                                                       train adapters
                                                            │
                                                       save adapter
                                                            │
                                                        evaluate
```

## 3. Configuration

Everything is in `configs/training.yaml` and mapped to
`cybersentinel.training.config.TrainingConfig`.

| Setting | Default | Why |
|---|---|---|
| `base_model_name` | `Qwen/Qwen2.5-3B-Instruct` | Strong instruction following at a size a consumer GPU can host. Configurable. |
| `max_seq_length` | 1024 | Events plus structured output fit comfortably; longer wastes memory. |
| `learning_rate` | 2e-4 | Standard for LoRA; far higher than full fine-tuning because only adapters move. |
| `num_train_epochs` | 3 | Enough for format and domain adherence on a corpus this size without memorising it. |
| `per_device_train_batch_size` | 2 | Fits alongside 4-bit weights and activations. |
| `gradient_accumulation_steps` | 8 | Effective batch size 16 without the memory of one. |
| `lora.r` / `lora.alpha` | 16 / 32 | The usual alpha = 2r ratio; rank 16 is sufficient for a format-and-domain adaptation. |
| `lora.dropout` | 0.05 | Light regularisation against a fairly regular corpus. |
| `lora.target_modules` | q,k,v,o,gate,up,down | Adapting MLP as well as attention lets a small rank change behaviour, not only attention routing. |
| `optim` | `paged_adamw_8bit` | Paged optimiser state survives memory spikes on a small card. |
| `gradient_checkpointing` | true | Trades compute for memory; the single biggest lever after quantisation. |
| `bf16` | true | Preferred over fp16 where supported: same memory, better numerical range. |

## 4. Hardware profiles

**Default profile** - a consumer NVIDIA GPU with roughly 12-16 GB.

**Low-memory profile** - `python scripts/train.py --low-memory`:
batch size 1, accumulation 16, sequence length 768, LoRA rank 8. Sequence length
and rank are the two changes that actually move peak memory once batch size is
already 1.

**CPU** - inference works (slowly); training does not, in any practical sense.
The script warns and continues rather than pretending otherwise.

**No GPU at all** - the application still runs. `LLM_BACKEND=mock` gives a
deterministic rules analyst, and the full pipeline, API, UI, tests and every
non-model evaluation work unchanged.

## 5. Running training

```bash
pip install -e ".[ml]"                       # torch, transformers, peft, trl, bitsandbytes
python scripts/prepare_dataset.py            # build the corpus first
python scripts/train.py --dry-run            # validate config and data, no GPU needed
python scripts/train.py                      # train
python scripts/train.py --low-memory         # smaller card
python scripts/train.py --epochs 5 --lora-r 32 --output models/run-b
```

`--dry-run` prints the resolved configuration, the detected hardware and any
problems found, and exits non-zero if the setup is not ready. Use it before
committing a GPU session.

Outputs land in `models/cybersentinel-lora/`:
`adapter_config.json`, `adapter_model.safetensors`, tokenizer files, and
`training_metrics.json` recording loss, runtime, hardware and the full
configuration used.

Note: `bitsandbytes` (4-bit quantisation) is Linux-only in the dependency
specification. On Windows, train under WSL2 or a Linux machine; the rest of the
project runs natively on Windows.

## 6. Serving the adapter

```bash
# .env
LLM_BACKEND=hf
MODEL_ADAPTER_PATH=models/cybersentinel-lora
```

Leave `MODEL_ADAPTER_PATH` empty to serve the base model - which is exactly how
the base arm of the evaluation is produced. `GET /health` reports which backend
and adapter are live.

## 7. Evaluation

Run the base and fine-tuned arms over the identical held-out test set:

```bash
python scripts/evaluate.py --with-hf --adapter models/cybersentinel-lora
```

Reported per arm: accuracy, precision, recall, macro F1, per-class metrics,
confusion matrix, JSON validity, field completeness, severity accuracy and
latency - on both the generated test set and the hard test set. See
`docs/evaluation.md`.

## 8. Honest expectations

On the **generated** test set there is little room to demonstrate a gain: the
corpus is template-derived and a keyword baseline already reaches 1.00 accuracy.
Expect all competent arms to cluster near the ceiling. That result is reported,
not hidden.

The **hard** test set is where the comparison is meaningful. It contains
paraphrased attacks with no signature vocabulary and benign events full of
alarming keywords; the rules baseline scores 0.11 there. Semantic understanding
is what separates the arms, and that is the number to report and defend.
