import json
import os
import re

from dotenv import load_dotenv
from groq import Groq
from langfuse import observe
from pydantic import BaseModel


class JudgeVerdict(BaseModel):
    passed: bool
    score: int  # 1-5
    rationale: str


# Neutral evaluator scaffolding. The styling judgment lives in the per-case
# rubric (authored in the golden set), not here. Do not encode styling rules
# in this template.
JUDGE_PROMPT_TEMPLATE = """You are an impartial evaluator. Score the stylist's top outfit pick for the occasion strictly against the rubric below. Judge only what the rubric asks. Do not re-rank, suggest other outfits, or add criteria of your own.

Occasion: {occasion}

Top outfit: {outfit}
Stylist reasoning: {reasoning}

Rubric:
{rubric}

Return ONLY a single JSON object, no markdown and no prose, matching exactly this shape:
{{"passed": true, "score": 3, "rationale": "<one or two sentences grounded in the rubric>"}}
where passed is a boolean, score is an integer from 1 to 5, and rationale is a string.
"""


def _extract_json_object(content: str | None) -> str:
    if not content or not content.strip():
        raise ValueError("Judge response was empty.")

    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Judge response did not contain a JSON object.")
    return text[start : end + 1]


def _parse_verdict(content: str) -> JudgeVerdict:
    return JudgeVerdict(**json.loads(_extract_json_object(content)))


ROUTING_REASON_PROMPT_TEMPLATE = """You are an impartial evaluator of a routing justification. A supervisor decided whether an outfit request needs external research about attire, and gave a reason. Judge ONLY whether the reason is a specific, correct justification for that decision.

A good reason identifies what attire is actually uncertain and why (when research was chosen), or correctly explains why the attire is already inferable from general knowledge (when research was skipped). A vague, circular, or incorrect reason fails. Research is not limited to formal events; cultural, themed, venue-specific, and activity-specific attire can also warrant it.

Occasion: {occasion}
Decision: {decision}
Supervisor's reason: {reason}

Return ONLY a single JSON object, no markdown:
{{"passed": true, "score": 3, "rationale": "<one or two sentences>"}}
"""


def _judge_with_retry(prompt: str, model: str) -> JudgeVerdict:
    client = Groq()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_completion_tokens=8192,
    )
    try:
        return _parse_verdict(response.choices[0].message.content)
    except Exception as first_error:
        retry = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You return only a single valid JSON object. No markdown."},
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\nThe previous response was not valid JSON "
                        f"because: {first_error}. Return ONLY the JSON object."
                    ),
                },
            ],
            temperature=0,
            max_completion_tokens=8192,
        )
        return _parse_verdict(retry.choices[0].message.content)


@observe()
def judge_routing_reason(occasion: str, decision: str, reason: str | None) -> JudgeVerdict:
    """Score whether the supervisor's routing reason is a specific, correct
    justification for its decision ('research' or 'skip'). Uses the configurable
    judge model so it does not grade its own output."""
    load_dotenv()
    model = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")
    prompt = ROUTING_REASON_PROMPT_TEMPLATE.format(
        occasion=occasion, decision=decision, reason=reason or "(none given)"
    )
    return _judge_with_retry(prompt, model)


@observe()
def judge_outfit(outfit: str, reasoning: str, occasion: str, rubric: str) -> JudgeVerdict:
    """
    LLM-as-judge for a single behavioral/color case. Scores the stylist's top
    outfit against the rubric and returns a structured JudgeVerdict.

    The judge model is configurable via JUDGE_MODEL (default
    openai/gpt-oss-120b) so the judge can differ from the stylist and avoid
    self-grading. One stricter retry on a bad parse; raises on persistent
    failure, no silent fallback.
    """
    load_dotenv()
    model = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        occasion=occasion, outfit=outfit, reasoning=reasoning, rubric=rubric
    )

    client = Groq()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_completion_tokens=8192,
    )
    content = response.choices[0].message.content

    try:
        return _parse_verdict(content)
    except Exception as first_error:
        retry = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You return only a single valid JSON object. No markdown.",
                },
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\nThe previous response was not valid JSON "
                        f"because: {first_error}. Return ONLY the JSON object."
                    ),
                },
            ],
            temperature=0,
            max_completion_tokens=8192,
        )
        return _parse_verdict(retry.choices[0].message.content)
