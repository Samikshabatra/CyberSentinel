"""Dataset construction for cybersecurity instruction tuning.

Pipeline:

    templates -> template-level split -> instance generation -> normalisation
              -> deduplication -> JSONL

Methodology notes (these are the parts that make the evaluation meaningful):

* **Template-level splitting.** Templates are assigned to train/validation/test
  before instances are generated. A phrasing that appears in training never
  appears in the test set, so test scores measure generalisation to unseen
  wording rather than memorised surface form.
* **Stratification.** Each category's templates are split independently, so
  every split covers every category.
* **Deterministic generation.** A fixed seed makes the corpus reproducible.
* **Deduplication.** Exact and normalised duplicates are removed within and
  across splits; any cross-split collision is dropped from the training side.
* **Provenance.** Every record records ``origin="synthetic"`` and its
  ``template_id``.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cybersentinel.cybersecurity.taxonomy import AttackType, Severity
from cybersentinel.llm.prompts import THREAT_DETECTION_INSTRUCTION, build_detection_messages
from cybersentinel.training.templates import TEMPLATES, EventTemplate, templates_by_category
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SEED = 20260808
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


@dataclass
class DatasetRecord:
    """One instruction-tuning example."""

    instruction: str
    input: str
    output: dict[str, Any]
    input_type: str = "alert"
    template_id: str = ""
    origin: str = "synthetic"
    label: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    def to_chat(self) -> dict[str, Any]:
        """Chat-format view used by the trainer.

        The prompt is built with the same template used at inference time, so
        training and serving never diverge.
        """
        messages = build_detection_messages(self.input, self.input_type)
        messages.append(
            {"role": "assistant", "content": json.dumps(self.output, ensure_ascii=False)}
        )
        return {"messages": messages}


@dataclass
class DatasetSplits:
    """Generated corpus with its documentation."""

    train: list[DatasetRecord] = field(default_factory=list)
    validation: list[DatasetRecord] = field(default_factory=list)
    test: list[DatasetRecord] = field(default_factory=list)
    card: dict[str, Any] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


#: Surface decorations applied to a portion of instances. They add the framing a
#: real SOC submission carries (a timestamp, a source system) without changing
#: the semantics of the event, and they expand the number of distinct instances
#: a low-slot template can produce.
_DECORATIONS: tuple[str, ...] = (
    "",
    "[{ts}] ",
    "{ts} SIEM alert: ",
    "Source: SIEM | Time: {ts} | ",
    "Ticket SOC-{ticket} ({ts}): ",
    "Detected at {ts}. ",
)

#: Effective distinct decorations, used when estimating instance capacity.
_DECORATION_CAPACITY = 500


def _timestamp(rng: random.Random) -> str:
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"2026-{month:02d}-{day:02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z"


def _fill(text: str, template: EventTemplate, rng: random.Random) -> tuple[str, dict[str, str]]:
    """Fill a template's slots with a deterministic random choice."""
    values = {name: rng.choice(options) for name, options in template.slots.items()}
    filled = text.format(**values) if values else text

    decoration = rng.choice(_DECORATIONS)
    if decoration:
        filled = (
            decoration.format(ts=_timestamp(rng), ticket=rng.randint(1000, 9999)) + filled
        )

    return filled, values


def _apply_values(items: Iterable[str], values: dict[str, str]) -> list[str]:
    return [item.format(**values) if values else item for item in items]


def _build_output(
    template: EventTemplate,
    values: dict[str, str],
    rng: random.Random,
) -> dict[str, Any]:
    """Build the target JSON for one example."""
    # Small confidence jitter so the model does not learn a constant.
    confidence = round(min(0.97, max(0.05, template.confidence + rng.uniform(-0.04, 0.04))), 2)

    output: dict[str, Any] = {
        "attack_type": template.attack_type.value,
        "severity": template.severity.value,
        "confidence": confidence,
        "evidence": _apply_values(template.evidence, values),
        "candidate_techniques": [template.technique] if template.technique else [],
        "reasoning": _reasoning(template, values),
    }
    return output


def _reasoning(template: EventTemplate, values: dict[str, str]) -> str:
    """Compose the reasoning field from the template's evidence."""
    if template.attack_type is AttackType.UNKNOWN:
        return (
            "The event does not contain enough detail to identify an attack technique. "
            "No classification is asserted; additional context is required."
        )
    evidence = _apply_values(template.evidence, values)
    joined = "; ".join(evidence[:2])
    if template.attack_type is AttackType.BENIGN:
        return (
            f"The observed activity matches expected behaviour ({joined}). "
            "No indicators of compromise are present."
        )
    return (
        f"The event shows {joined}. Taken together these are consistent with "
        f"{template.attack_type.value}, at {template.severity.value} severity."
    )


def split_templates(
    seed: int = DEFAULT_SEED,
    ratios: dict[str, float] | None = None,
) -> dict[str, list[EventTemplate]]:
    """Assign templates to splits, stratified by category.

    Every category with at least three templates contributes to all three
    splits. Categories with fewer templates fill train first, then test - a
    category is never left absent from the test set when it has two or more.
    """
    resolved = ratios or SPLIT_RATIOS
    rng = random.Random(seed)
    assignment: dict[str, list[EventTemplate]] = {"train": [], "validation": [], "test": []}

    for category, templates in templates_by_category().items():
        shuffled = list(templates)
        rng.shuffle(shuffled)
        total = len(shuffled)

        if total == 1:
            assignment["train"].extend(shuffled)
            logger.warning(
                f"category '{category.value}' has a single template; it appears in train only"
            )
            continue
        if total == 2:
            assignment["train"].append(shuffled[0])
            assignment["test"].append(shuffled[1])
            continue

        n_test = max(1, round(total * resolved["test"]))
        n_validation = max(1, round(total * resolved["validation"]))
        if n_test + n_validation >= total:
            n_test, n_validation = 1, 1

        assignment["test"].extend(shuffled[:n_test])
        assignment["validation"].extend(shuffled[n_test : n_test + n_validation])
        assignment["train"].extend(shuffled[n_test + n_validation :])

    return assignment


def _slot_combinations(template: EventTemplate) -> int:
    """Number of distinct instances a template can produce.

    Slot choices multiply with the surface decorations (timestamps, source
    tags), so even a template with a single slot can yield many distinct texts.
    """
    combinations = 1
    for options in template.slots.values():
        combinations *= len(options)
    return max(1, combinations) * _DECORATION_CAPACITY


def generate_records(
    templates: list[EventTemplate],
    instances_per_template: int,
    seed: int,
    target_per_category: int | None = None,
) -> list[DatasetRecord]:
    """Generate instances from a set of templates.

    When ``target_per_category`` is given, the per-template instance count is
    derived from it. This is what keeps the split ratios at 80/10/10 by
    *instance* count even though templates are split at template level: a
    category whose test split holds one template still yields the same number of
    test instances as one whose test split holds three.
    """
    rng = random.Random(seed)
    records: list[DatasetRecord] = []

    by_category: dict[AttackType, list[EventTemplate]] = {}
    for template in templates:
        by_category.setdefault(template.attack_type, []).append(template)

    for category_templates in by_category.values():
        if target_per_category:
            per_template = max(1, -(-target_per_category // len(category_templates)))
        else:
            per_template = instances_per_template

        for template in category_templates:
            # A template with few slots cannot produce unlimited distinct text.
            count = min(per_template, _slot_combinations(template))
            for _ in range(count):
                text, values = _fill(template.text, template, rng)
                records.append(
                    DatasetRecord(
                        instruction=THREAT_DETECTION_INSTRUCTION,
                        input=text,
                        output=_build_output(template, values, rng),
                        input_type=template.input_type,
                        template_id=template.template_id,
                        label=template.attack_type.value,
                    )
                )

    return records


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def deduplicate(
    records: list[DatasetRecord],
    seen: set[str] | None = None,
) -> tuple[list[DatasetRecord], int]:
    """Remove records whose normalised input was already seen."""
    known = seen if seen is not None else set()
    unique: list[DatasetRecord] = []
    removed = 0

    for record in records:
        key = _normalise(record.input)
        if key in known:
            removed += 1
            continue
        known.add(key)
        unique.append(record)

    return unique, removed


def class_distribution(records: list[DatasetRecord]) -> dict[str, int]:
    return dict(Counter(record.label for record in records))


def build_dataset(
    examples_per_category: int = 120,
    seed: int = DEFAULT_SEED,
    ratios: dict[str, float] | None = None,
) -> DatasetSplits:
    """Build the full corpus with train/validation/test splits.

    ``examples_per_category`` is the target number of examples per attack
    category across all splits, divided between them by ``ratios``. Balancing by
    category is deliberate: real incident frequencies are wildly skewed, and
    training on that skew would teach the model the prior rather than the
    evidence. The imbalance is reintroduced only where it belongs - in how the
    evaluation reports per-class metrics.
    """
    resolved = ratios or SPLIT_RATIOS
    assignment = split_templates(seed=seed, ratios=resolved)

    targets = {
        name: max(1, round(examples_per_category * ratio)) for name, ratio in resolved.items()
    }

    # Test and validation are deduplicated first, so any collision is removed
    # from training rather than from the evaluation sets.
    seen: set[str] = set()
    test, test_removed = deduplicate(
        generate_records(assignment["test"], 0, seed + 3, targets["test"]), seen
    )
    validation, validation_removed = deduplicate(
        generate_records(assignment["validation"], 0, seed + 2, targets["validation"]), seen
    )
    train, train_removed = deduplicate(
        generate_records(assignment["train"], 0, seed + 1, targets["train"]), seen
    )

    splits = DatasetSplits(train=train, validation=validation, test=test)
    splits.card = {
        "name": "CyberSentinel synthetic cybersecurity instruction dataset",
        "version": "1.0",
        "origin": "synthetic",
        "generator": "cybersentinel.training.dataset",
        "seed": seed,
        "examples_per_category_target": examples_per_category,
        "split_targets_per_category": targets,
        "templates_total": len(TEMPLATES),
        "templates_per_split": {name: len(items) for name, items in assignment.items()},
        "template_ids_per_split": {
            name: sorted(template.template_id for template in items)
            for name, items in assignment.items()
        },
        "counts": splits.counts(),
        "class_distribution": {
            "train": class_distribution(train),
            "validation": class_distribution(validation),
            "test": class_distribution(test),
        },
        "duplicates_removed": {
            "train": train_removed,
            "validation": validation_removed,
            "test": test_removed,
        },
        "split_method": (
            "Template-level, stratified by attack category. Templates are assigned to a split "
            "before instances are generated, so no phrasing is shared between train and test. "
            "Instance counts per split are then scaled to the 80/10/10 ratio, because a "
            "category may hold a different number of templates in each split."
        ),
        "contamination_controls": [
            "Templates are partitioned between splits; instances never cross splits.",
            "Normalised-text deduplication is applied test-first, then validation, then train.",
            "Test examples were not used to author training examples.",
        ],
        "slot_data_policy": (
            "Addresses come from RFC 5737 documentation ranges and domains from RFC 2606 "
            "example domains. No real infrastructure, person or organisation is referenced."
        ),
        "licence": "Generated for this project; released with the project under the MIT licence.",
        "limitations": [
            "Template-generated text is more regular than production telemetry.",
            "Category priors reflect template counts, not real-world incident frequency.",
            "Severity labels follow the project's own convention, not a published standard.",
        ],
    }
    return splits


def write_splits(splits: DatasetSplits, output_dir: Path) -> dict[str, Path]:
    """Write JSONL splits, a chat-format copy for training, and the dataset card."""
    paths: dict[str, Path] = {}

    for name in ("train", "validation", "test"):
        records: list[DatasetRecord] = getattr(splits, name)
        directory = output_dir / name
        directory.mkdir(parents=True, exist_ok=True)

        raw_path = directory / f"{name}.jsonl"
        raw_path.write_text(
            "\n".join(record.to_json() for record in records) + "\n", encoding="utf-8"
        )

        chat_path = directory / f"{name}_chat.jsonl"
        chat_path.write_text(
            "\n".join(json.dumps(record.to_chat(), ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )

        paths[name] = raw_path

    card_path = output_dir / "dataset_card.json"
    card_path.write_text(json.dumps(splits.card, indent=2), encoding="utf-8")
    paths["card"] = card_path
    return paths


def load_records(path: Path) -> list[DatasetRecord]:
    """Load a JSONL split back into records."""
    records: list[DatasetRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            records.append(
                DatasetRecord(
                    instruction=data["instruction"],
                    input=data["input"],
                    output=data["output"],
                    input_type=data.get("input_type", "alert"),
                    template_id=data.get("template_id", ""),
                    origin=data.get("origin", "synthetic"),
                    label=data.get("label", data["output"].get("attack_type", "")),
                )
            )
    return records


def severity_distribution(records: list[DatasetRecord]) -> dict[str, int]:
    return dict(Counter(str(record.output.get("severity", Severity.UNKNOWN.value)) for record in records))
