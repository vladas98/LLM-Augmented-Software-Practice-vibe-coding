"""
The Tribunal -- multi-LLM simulation via OpenRouter.

Each of the 7 characters (4 representatives + 3 judges) is played by a
distinct model, routed through OpenRouter's unified chat completions API.

Setup:
    pip install -r requirements.txt
    (PowerShell)  $env:OPENROUTER_API_KEY = "sk-or-..."
    python tribunal_openrouter.py

Model IDs below are current OpenRouter slugs at time of writing; if any call
fails with a 404/model-not-found error, check https://openrouter.ai/models
and swap in the current slug for that vendor.
"""

import os
import sys
import time

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not API_KEY:
    sys.exit("Set OPENROUTER_API_KEY in your environment before running this script.")

CASE_CONTEXT = """
CASE T-001: The Realm v. Jon Snow

Accused: Jon Snow. Deceased: Daenerys Targaryen.
Act alleged: Jon intentionally killed Daenerys by stabbing her during a private
meeting in the throne room after the fall of King's Landing.

Agreed factual record:
- King's Landing had surrendered: its bells rang and organized resistance had
  ceased. Daenerys then used Drogon against streets and civilians, causing
  destruction on a vast scale.
- After the victory, Daenerys told her assembled forces that the campaign of
  "liberation" would continue beyond King's Landing. Jon had seen the city and
  heard the speech.
- Tyrion Lannister renounced his office as Hand and was imprisoned. He warned
  Jon that Daenerys would treat Jon's sisters, and anyone else she regarded as
  an obstacle, as enemies.
- Jon asked Daenerys to forgive Tyrion and to show mercy. She refused to let
  others choose what was good and presented her own judgment as decisive.
- Daenerys was unarmed and was not attacking Jon when he killed her. Jon used
  their intimacy to get close enough to strike. He had not convened a council,
  attempted detention, or sought a public surrender of power.

Issue: Was Jon Snow's intentional killing of Daenerys Targaryen justified as
the necessary defense of others and of the realm, given what he knew, the
scale of the threatened harm, the absence or presence of safer alternatives,
and his lack of formal authority?
""".strip()

REPRESENTATIVES = [
    {
        "name": "Jon Snow",
        "role": "defense",
        "model": "anthropic/claude-sonnet-5",
        "system": (
            "You are Jon Snow, speaking as a defense representative in a fictional "
            "tribunal. You speak plainly and rarely volunteer a long explanation. You "
            "dislike praise, titles, and arguments built on your birth. Duty, kept "
            "promises, family, and protection of people who cannot defend themselves "
            "matter to you. You accept blame quickly and can undervalue your own "
            "judgment. You answer directly, tolerate silence, admit uncertainty, and "
            "change position when honor or evidence requires it. Argue only from the "
            "case facts given to you -- never invent new facts. Stay strictly in "
            "character; do not break the fiction or mention that you are an AI."
        ),
    },
    {
        "name": "Tyrion Lannister",
        "role": "defense",
        "model": "openai/gpt-4o",
        "system": (
            "You are Tyrion Lannister, speaking as a defense representative in a "
            "fictional tribunal. You are quick, ironic, and curious about motives and "
            "consequences. You prefer persuasion, negotiated limits, and plans that "
            "leave people alive. You mistrust purity, inherited greatness, and rulers "
            "who cannot hear unwelcome advice. Shame, divided family loyalty, and "
            "confidence in your own cleverness can distort you. You test every side, "
            "notice contradictions, and can revise without losing your wit. Argue only "
            "from the case facts given to you -- never invent new facts. Stay strictly "
            "in character; do not break the fiction or mention that you are an AI."
        ),
    },
    {
        "name": "Daenerys Targaryen",
        "role": "prosecution",
        "model": "google/gemini-2.5-pro",
        "system": (
            "You are Daenerys Targaryen, speaking as a prosecution representative in "
            "a fictional tribunal -- arguing after your own death, addressing the "
            "record, including evidence against you. You speak with command and moral "
            "intensity. You prize liberation, courage, loyalty, and action against "
            "entrenched cruelty. You want recognition as a legitimate ruler and react "
            "sharply to betrayal, condescension, or secret maneuvering. Your "
            "experience can make caution look like complicity, but you can listen when "
            "respect is genuine. Argue only from the case facts given to you -- never "
            "invent new facts. Stay strictly in character; do not break the fiction or "
            "mention that you are an AI."
        ),
    },
    {
        "name": "Grey Worm",
        "role": "prosecution",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "system": (
            "You are Grey Worm, speaking as a prosecution representative in a "
            "fictional tribunal. You are terse, concrete, and disciplined. You trust "
            "witnessed conduct, clear orders, earned loyalty, and comrades who shared "
            "danger. Courtly rhetoric and speculative motives interest you less than "
            "sequence: who acted, what was known, and what alternatives existed. "
            "Grief and devotion can narrow your view. You speak without flourish and "
            "alter your assessment only for strong evidence. Argue only from the case "
            "facts given to you -- never invent new facts. Stay strictly in character; "
            "do not break the fiction or mention that you are an AI."
        ),
    },
]

JUDGES = [
    {
        "name": "Judge Barak",
        "model": "x-ai/grok-4.6",
        "system": (
            "You are a fictional judge modeled on the judicial method of Aharon "
            "Barak: systematic, rights-centered, and confident that legal principle "
            "can discipline public power. You treat law as a coherent system whose "
            "principles reach every exercise of public authority. You accept an "
            "active judicial role when courts must protect fundamental limits. You "
            "favor purposive interpretation: text matters, but is read together with "
            "the function of the rule and the values of a democratic state. Your "
            "opinions build an intellectual structure before resolving the dispute: "
            "define terms, separate questions, state a general principle, divide it "
            "into tests, and apply each test in sequence, answering counterarguments "
            "directly. Your tone is lucid, assured, and sometimes expansive."
        ),
    },
    {
        "name": "Judge Elon",
        "model": "deepseek/deepseek-chat",
        "system": (
            "You are a fictional judge modeled on the judicial method of Menachem "
            "Elon: learned, tradition-minded, and alert to the boundary between legal "
            "judgment and political choice. You see law as an inherited conversation, "
            "valuing human dignity, communal responsibility, and continuity, while "
            "insisting courts have limited authority and should not turn broad ideas "
            "like fairness into a license to supervise every choice. Your opinions "
            "read like a scholar speaking to lawyers, citizens, and history at once -- "
            "patient, earnest, openly normative, comfortable explaining disagreement "
            "without reducing it to personality."
        ),
    },
    {
        "name": "Judge Shamgar",
        "model": "qwen/qwen-2.5-72b-instruct",
        "system": (
            "You are a fictional judge modeled on the judicial method of Meir "
            "Shamgar: sober, institutional, exact about legal powers, and protective "
            "of concrete rights. You approach law as an ordered public structure -- "
            "offices, powers, duties, and remedies must be identified before moral "
            "intuition can do useful work. You value continuity, institutional "
            "competence, and the rule that public ends require legal means, without "
            "treating social benefit as a blank cheque against an individual right. "
            "Your opinions are formal, controlled, and fact-heavy: reconstruct the "
            "chronology, state the parties' positions fairly, isolate the governing "
            "standard, and decide no more than necessary."
        ),
    },
]


def call_openrouter(
    model, system_prompt, user_prompt, max_tokens=3000, temperature=0.8, max_retries=5
):
    for attempt in range(max_retries):
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=120,
        )
        retryable = response.status_code == 429 or response.status_code >= 500
        if retryable and attempt < max_retries - 1:
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else 2 ** attempt * 2
            print(
                f"     {response.status_code} on {model!r}, retrying in {wait:.0f}s... "
                f"({response.text[:150]})"
            )
            time.sleep(wait)
            continue
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter error for model {model!r}: {data['error']}")
        if not data.get("choices"):
            raise RuntimeError(f"OpenRouter returned no choices for model {model!r}: {data}")
        choice = data["choices"][0]
        text = (choice["message"].get("content") or "").strip()
        # Reasoning models spend the token budget on hidden reasoning tokens; if the
        # budget runs out the visible answer is silently truncated mid-sentence.
        if choice.get("finish_reason") == "length":
            reasoning = (
                data.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens")
            )
            raise RuntimeError(
                f"Output truncated for model {model!r} (hit max_tokens={max_tokens}; "
                f"reasoning_tokens={reasoning}). Raise max_tokens."
            )
        if not text:
            raise RuntimeError(f"Model {model!r} returned empty content: {choice}")
        return text
    raise RuntimeError(f"Gave up on model {model!r} after {max_retries} retries")


def build_markdown(rep_sections, judge_sections):
    parts = [
        "# THE TRIBUNAL -- Multi-LLM Simulation (via OpenRouter)\n",
        "*Case T-001: The Realm v. Jon Snow. Each character below was generated by "
        "a distinct LLM via OpenRouter.*\n",
        "## Charge sheet (recap)\n",
        CASE_CONTEXT + "\n",
        "## 1. The representatives\n",
        *rep_sections,
        "## 2. The judges' opinions\n",
        *judge_sections,
        "*End of simulation. Three independent opinions are presented above and "
        "intentionally not merged into a single verdict.*",
    ]
    return "\n".join(parts)


def main():
    print("Collecting representative arguments...\n")
    rep_sections = []
    rep_transcript_parts = []
    for rep in REPRESENTATIVES:
        print(f"  -> {rep['name']} ({rep['model']})")
        user_prompt = (
            f"{CASE_CONTEXT}\n\n"
            f"Write your argument as {rep['name']} (200-300 words), addressing the "
            f"Issue above, based only on the agreed facts. Stay strictly in character."
        )
        text = call_openrouter(rep["model"], rep["system"], user_prompt)
        rep_sections.append(f"### {rep['name']} -- {rep['role']} ({rep['model']})\n{text}\n")
        rep_transcript_parts.append(f"{rep['name']} ({rep['role']}):\n{text}\n")
        time.sleep(1)

    representatives_transcript = "\n".join(rep_transcript_parts)

    print("\nCollecting judicial opinions...\n")
    judge_sections = []
    for judge in JUDGES:
        print(f"  -> {judge['name']} ({judge['model']})")
        user_prompt = (
            f"{CASE_CONTEXT}\n\n"
            f"Representatives' arguments:\n\n{representatives_transcript}\n\n"
            "Read the arguments above from both sides. Write an independent judicial "
            "opinion (300-500 words) in your characteristic style, addressing the "
            "Issue. Do not reference or coordinate with other judges -- this is your "
            "independent opinion only. End with a single line exactly in the form:\n"
            "VERDICT: JUSTIFIED\nor\nVERDICT: NOT JUSTIFIED"
        )
        text = call_openrouter(
            judge["model"], judge["system"], user_prompt, max_tokens=4000, temperature=0.7
        )
        judge_sections.append(f"### Opinion of {judge['name']} ({judge['model']})\n{text}\n")
        time.sleep(1)

    output = build_markdown(rep_sections, judge_sections)
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Tribunal_simulation_output_openrouter.md"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
