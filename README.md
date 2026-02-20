# Slop Guard Plugin for Data Designer

A [Data Designer](https://github.com/NVIDIA/NeMo-Data-Designer) plugin that scores text columns for formulaic AI writing patterns using regex-based analysis. No LLM calls, no API dependencies — pure regex.

Runs ~80 compiled patterns against text and returns a numeric score (0–100), a severity band, and actionable advice for each violation. Core analysis engine extracted from [eric-tramel/slop-guard](https://github.com/eric-tramel/slop-guard) (MIT).

## Install

Requires `data-designer` to be installed in your environment.

```bash
# From a project that already has data-designer installed
pip install data-designer-slop-guard

# Or with uv
uv pip install data-designer-slop-guard

# From source (for development)
git clone https://github.com/kirit93/data-designer-slop-guard.git
pip install -e data-designer-slop-guard
```

The plugin registers itself via Python entry points. Data Designer's `PluginRegistry` auto-discovers it on next import — no core code changes needed. Just install and use.

## Usage

### Score a text column

```python
from data_designer_slop_guard import SlopGuardColumnConfig

builder.add_column(SlopGuardColumnConfig(
    name="slop_check",
    target_columns=["article"],
    min_score=60,
))
```

Output per row:

```json
{
  "is_valid": false,
  "slop_score": 42,
  "slop_band": "moderate",
  "word_count": 312,
  "slop_advice": [
    "Replace 'delve' — what specifically do you mean?",
    "Cut 'it's worth noting' — just state the point directly."
  ]
}
```

### Score + Improve (three-column pattern)

```python
import data_designer.config as dd
from data_designer_slop_guard import SlopGuardColumnConfig

# 1. Generate
builder.add_column(dd.LLMTextColumnConfig(
    name="story",
    prompt="Write a short story about {{ topic }}",
    model_alias="writer",
))

# 2. Score
builder.add_column(SlopGuardColumnConfig(
    name="slop_check",
    target_columns=["story"],
    min_score=60,
    include_advice=True,
))

# 3. Improve using advice
builder.add_column(dd.LLMTextColumnConfig(
    name="story_improved",
    prompt="""Rewrite this text to remove AI writing patterns.

Original: {{ story }}

Score: {{ slop_check.slop_score }}/100 ({{ slop_check.slop_band }})
Issues:
{% for tip in slop_check.slop_advice %}- {{ tip }}
{% endfor %}

Return only the improved text.""",
    model_alias="writer",
))
```

Data Designer's DAG resolver runs them in order: `story` → `slop_check` → `story_improved`.

## Config options

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target_columns` | `list[str]` | required | Text columns to score (concatenated per row) |
| `min_score` | `int` | `60` | Score threshold for `is_valid=True` |
| `include_advice` | `bool` | `True` | Include actionable advice strings |
| `include_violations` | `bool` | `False` | Include raw violation details |

## Score bands

| Score | Band |
|---|---|
| 80–100 | clean |
| 60–79 | light |
| 40–59 | moderate |
| 20–39 | heavy |
| 0–19 | saturated |

## What it catches

Slop words (adjectives, verbs, nouns, hedging adverbs), stock phrases and filler, structural patterns (bold-header blocks, bullet runs, triadic lists), tone markers (meta-communication, false narrativity, sentence-opener tells), weasel phrases, AI self-disclosure, placeholder text, rhythm monotony, em dash density, contrast pairs, setup-resolution flips, elaboration colon density, pithy fragments, and repeated n-grams.

## Standalone usage

The core analysis function works independently of Data Designer:

```python
from data_designer_slop_guard import analyze_text

result = analyze_text("Your text here...")
print(result["score"], result["band"], result["advice"])
```

## Tests

```bash
uv run --with pytest pytest tests/ -v
```

## Acknowledgments

The core analysis engine in this plugin is extracted from [**slop-guard**](https://github.com/eric-tramel/slop-guard) by [Eric Tramel](https://github.com/eric-tramel). Eric built the original ~80-rule regex linter and the scoring model (exponential decay with concentration multipliers for Claude-specific patterns). His [blog post](https://eric-tramel.github.io/blog/2026-02-18-slop-guard/) covers the design rationale. The original code is released under the MIT license.

This plugin wraps that engine as a Data Designer column generator plugin so it can be used to score and improve synthetic datasets at scale.

## License

Apache-2.0. Core analysis engine from [eric-tramel/slop-guard](https://github.com/eric-tramel/slop-guard) under MIT license.
