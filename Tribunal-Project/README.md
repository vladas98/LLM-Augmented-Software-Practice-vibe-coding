# The Tribunal — Multi-LLM Simulation

A multi-agent simulation of a fictional legal proceeding, where **each character is played by a different LLM** routed through [OpenRouter](https://openrouter.ai).

## The case

**T-001: The Realm v. Jon Snow** — Jon Snow intentionally killed Daenerys Targaryen after the fall of King's Landing.

> **Issue:** Was the killing justified as the necessary defense of others and of the realm, given what he knew, the scale of the threatened harm, the absence or presence of safer alternatives, and his lack of formal authority?

Four representatives argue the case, then three judges — each modeled on the judicial method of a real Israeli Supreme Court justice — write **independent** opinions. The opinions are deliberately *not* merged into a single verdict.

## Model assignment

Seven distinct models across seven vendors, so the reasoning diversity is real rather than one model imitating seven voices:

| Character | Seat | Model |
|---|---|---|
| Jon Snow | defense | `anthropic/claude-sonnet-5` |
| Tyrion Lannister | defense | `openai/gpt-4o` |
| Daenerys Targaryen | prosecution | `google/gemini-2.5-pro` |
| Grey Worm | prosecution | `meta-llama/llama-3.3-70b-instruct` |
| Judge Barak | bench | `x-ai/grok-4.6` |
| Judge Elon | bench | `deepseek/deepseek-chat` |
| Judge Shamgar | bench | `qwen/qwen-2.5-72b-instruct` |

Each persona's system prompt encodes its character and reasoning style; representatives argue only from the agreed factual record. Judges receive the full transcript of all four arguments and are instructed to reason independently.

## Setup

```bash
pip install -r requirements.txt
```

Set your OpenRouter API key as an environment variable — it is **never** stored in the code:

```powershell
# PowerShell (current session)
$env:OPENROUTER_API_KEY = "sk-or-..."
```

```bash
# bash
export OPENROUTER_API_KEY="sk-or-..."
```

## Run

```bash
python tribunal_openrouter.py
```

Output is written to `Tribunal_simulation_output_openrouter.md`.

## Implementation notes

A few things this script handles that are easy to get wrong:

- **Retries with backoff** on `429` and `5xx` responses, honoring `Retry-After` when present. Upstream providers rate-limit and time out intermittently.
- **Provider errors inside HTTP 200 bodies** — OpenRouter can return `{"error": ...}` with a success status, so the response is validated rather than indexed blindly.
- **Truncation guard.** Reasoning models spend the `max_tokens` budget on hidden reasoning tokens before emitting visible text. At `max_tokens=700`, `gemini-2.5-pro` used 669 tokens on reasoning and returned a single truncated clause. The script now raises on `finish_reason == "length"` instead of silently writing a half-sentence into the transcript.

## Note on results

In the committed run, all three judges reached **JUSTIFIED** — a convergence worth examining rather than assuming, given that the three judicial models were built to reason differently.

## Disclaimer

A fictional proceeding. The judge profiles adapt documented judicial *methods*; they do not impersonate the judges or predict how any real court would rule.
