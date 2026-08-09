# Dataset

## 1. What this dataset is

A synthetic cybersecurity instruction-tuning corpus, generated from hand-written
event templates. It is **fully synthetic and labelled as such** in every record
(`origin: "synthetic"`). No real incident data, customer data or personal data is
used.

Build it with:

```bash
python scripts/prepare_dataset.py            # defaults: 120 examples per category
python scripts/prepare_dataset.py --per-category 200 --seed 7
```

Outputs:

```
data/train/train.jsonl            data/train/train_chat.jsonl
data/validation/validation.jsonl  data/validation/validation_chat.jsonl
data/test/test.jsonl              data/test/test_chat.jsonl
data/test/hard_test.jsonl         (hand-authored, not generated)
data/dataset_card.json            provenance and distribution
```

## 2. Why synthetic

Public cybersecurity corpora exist, but the ones with permissive licences are
mostly network-flow features or malware hashes, not natural-language events with
structured analyst conclusions. Downloading an unrelated dataset and calling it
cybersecurity instruction tuning would not be defensible. Generating a corpus
from documented attack patterns, with the evidence a correct analysis should
extract written alongside each label, gives:

- exact control over the label distribution,
- an evidence field to train on, not just a class,
- a documented, reproducible provenance chain.

The cost is realism, which is stated plainly in the limitations below.

## 3. Record schema

```json
{
  "instruction": "Analyze the following cybersecurity event.",
  "input": "47 failed SSH login attempts from 198.51.100.23 within 3 minutes...",
  "output": {
    "attack_type": "Brute Force",
    "severity": "HIGH",
    "confidence": 0.92,
    "evidence": [
      "47 failed authentication attempts in a 3 minute window",
      "all attempts originate from the single source 198.51.100.23",
      "attempts concentrate on one account: root"
    ],
    "candidate_techniques": ["T1110"],
    "reasoning": "The event shows ... consistent with Brute Force, at HIGH severity."
  },
  "input_type": "alert",
  "template_id": "bf-01",
  "origin": "synthetic",
  "label": "Brute Force"
}
```

`*_chat.jsonl` contains the same records rendered into the chat format used at
inference time, so training and serving share one prompt definition
(`llm/prompts.py`).

## 4. Categories

All 15 taxonomy labels are represented: Phishing, Brute Force, Credential
Attack, SQL Injection, Cross-Site Scripting, Malware, DDoS / Network Attack,
Port Scanning / Reconnaissance, Privilege Escalation, Data Exfiltration,
Suspicious Authentication Activity, Vulnerability / Exploit, Insider Threat,
Benign, Unknown.

`Unknown` is trained deliberately. Templates `unk-01` to `unk-04` contain inputs
with no actionable evidence and teach the model to decline. Without them the
model learns that every input must receive a label, which is exactly the
behaviour the hallucination evaluation penalises.

## 5. Splitting methodology

**Template-level, stratified, before generation.**

1. Templates are grouped by attack category.
2. Each category's templates are shuffled with a fixed seed and partitioned
   80/10/10 into train, validation and test.
3. Only then are instances generated, filling slots and applying surface
   decorations (timestamps, ticket references, source tags).
4. Instance counts per split are scaled to the 80/10/10 ratio, because a
   category may hold a different number of templates in each split.

The consequence: **a phrasing seen in training never appears in the test set.**
Splitting instances instead would leak wording across the boundary and inflate
the reported scores.

Categories with only two templates contribute to train and test but not
validation; this is recorded in the dataset card.

## 6. Contamination controls

- Templates are partitioned between splits before instance generation.
- Normalised-text deduplication runs **test first, then validation, then
  train**, so a collision is removed from training rather than from an
  evaluation set.
- Test examples were not used to author training examples.
- The dataset card lists the exact template ids assigned to each split.

## 7. Class distribution

Balanced by construction: each category targets the same number of examples.

Real incident frequencies are wildly skewed (reconnaissance is constant,
insider threats are rare). Training on that skew would teach the model the prior
rather than the evidence, and a model that predicts the base rate scores well on
accuracy while being useless. The skew is reintroduced where it belongs: the
evaluation reports per-class metrics and uses macro F1 as the headline figure.

A representative build (`--per-category 120`) produces roughly:

| Split | Examples | Categories |
|---|---|---|
| train | ~1,330 | 15 |
| validation | ~160 | 14 |
| test | ~175 | 15 |

Exact numbers for any build are in `data/dataset_card.json`.

## 8. The hard test set

`data/test/hard_test.jsonl` is hand-authored, not generated, and contains three
kinds of case:

- **Paraphrases** - real attacks described in plain language with no signature
  vocabulary ("the same address kept trying one of our accounts over and over").
- **Keyword decoys** - benign events packed with alarming words: an authorised
  penetration test, a scheduled vulnerability scan, a `DROP TABLE` in an
  approved maintenance window, a phishing-awareness training reminder.
- **Insufficient evidence** - inputs where the correct answer is `Unknown`.

This set exists because the generated test set is drawn from the same grammar as
the training data, and a keyword system can therefore saturate it. Measured on
the current build, the rules baseline scores **1.00 accuracy on the generated
test set and 0.11 on the hard set**. That gap is the honest measure of how much
of the generated benchmark is surface pattern matching, and it is the set on
which a fine-tuned model has room to demonstrate benefit.

## 9. Slot data policy

- IPv4 addresses come from the RFC 5737 documentation ranges (192.0.2.0/24,
  198.51.100.0/24, 203.0.113.0/24).
- Domains come from RFC 2606 reserved example domains.
- Usernames, hostnames and hashes are invented placeholders.

No real infrastructure, person or organisation is referenced anywhere.

## 10. Licence

Generated by this project and released with it under the MIT licence. The
knowledge base used by RAG is separate and carries its own source attributions
(see `docs/rag.md`).

## 11. Limitations

- Template-generated text is more regular than production telemetry; real logs
  are noisier, truncated and inconsistently formatted.
- Category priors reflect template counts, not real-world incident frequency.
- Severity labels follow this project's convention, not a published standard.
- Evidence strings are authored alongside the template, so they are cleaner and
  more complete than what an analyst would extract under time pressure.
- The corpus is English-only.

## 12. Extending the dataset

Add an `EventTemplate` to `src/cybersentinel/training/templates.py` with its
category, severity, text (with `{slot}` placeholders), the evidence a correct
analysis should extract, and a candidate technique id that exists in the
verified catalogue. Regenerate with `scripts/prepare_dataset.py`. The splitter
will place the new template automatically and the dataset card will record it.
