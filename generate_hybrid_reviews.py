"""Generate hybrid review data for description-conditioned answer prediction."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DEFAULT_OUTPUT_PATH = Path("hybrid_review_dataset.json")
NUM_DESCRIPTIONS = 100
QUESTION_POOL_SIZE = 100
QUESTIONS_PER_DESCRIPTION = 10

ITEM_TYPES = ["paragraph", "ad", "video", "campaign", "project"]
AUDIENCES = ["developers", "parents", "students", "small-business owners", "donors"]
TONES = ["playful", "measured", "urgent", "luxury", "grassroots"]
NOVELTY_LEVELS = ["safe", "balanced", "experimental"]
EVIDENCE_LEVELS = ["light", "medium", "heavy"]
CTA_STYLES = ["none", "soft", "hard"]
PACE_LEVELS = ["slow", "steady", "fast"]
VISUAL_STYLES = ["minimal", "polished", "chaotic"]
VALUES = ["clarity", "novelty", "credibility", "emotion", "efficiency"]


@dataclass(frozen=True)
class HybridProfile:
    name: str
    description: str
    preferred_item_type: str
    audience: str
    tone: str
    novelty: str
    evidence: str
    cta_style: str
    pace: str
    visual_style: str
    value: str
    risk_tolerance: str


@dataclass(frozen=True)
class ReviewPrompt:
    question_id: int
    item_type: str
    audience: str
    tone: str
    novelty: str
    evidence: str
    cta_style: str
    pace: str
    visual_style: str
    value: str
    paragraph: str

    def render(self) -> str:
        return f"What do you think of this {self.item_type}?\n\n{self.paragraph}"


def vader_label(compound: float) -> str:
    if compound >= 0.65:
        return "strong_like"
    if compound >= 0.25:
        return "like"
    if compound <= -0.65:
        return "strong_dislike"
    if compound <= -0.25:
        return "dislike"
    return "neutral"


# Selfish/negative "hater" archetypes - VERY different from each other
HATER_PRIMARY_ROLES = [
    "troll who enjoys provoking and does not care about substance",
    "cynic who assumes everything is a scam and trusts nothing",
    "gatekeeper who excludes and dismisses as not real or legitimate",
    "envious critic who tears down what they secretly covet",
    "nitpicker who obsesses over tiny flaws and misses the whole point",
    "armchair expert who condescends and knows better than everyone",
    "grudge holder who brings unrelated baggage and cannot evaluate fairly",
    "dismissive skeptic who has seen it all and nothing impresses",
    "contrarian who disagrees for sport regardless of merit",
    "self-absorbed reviewer who only cares how it affects them personally",
]
HATER_SECONDARY_ROLES = [
    "fault-finder who looks for problems first and celebrates finding them",
    "tear-down enthusiast who enjoys criticizing more than understanding",
    "never-satisfied perfectionist with impossible standards",
    "superiority-complex type who talks down to everyone",
    "whataboutist who deflects with tangents and false equivalences",
    "straw-man builder who misrepresents ideas to attack them",
    "ad-hominem leaner who attacks the creator instead of the work",
    "goalpost mover who changes criteria mid-review to keep disliking",
    "sealion who bad-faith just-asking-questions to exhaust others",
    "bad-faith reader who assumes the worst interpretation always",
]


def build_hybrid_profiles(n: int, seed: int) -> list[HybridProfile]:
    rng = random.Random(seed)
    primary_roles = [
        "brand strategist",
        "performance marketer",
        "creative producer",
        "product storyteller",
        "community builder",
        "operations lead",
        "user researcher",
        "growth analyst",
        "design critic",
        "campaign planner",
    ]
    secondary_roles = [
        "skeptical editor",
        "empathetic operator",
        "taste-driven builder",
        "metrics-first reviewer",
        "risk-aware optimist",
        "detail-obsessed finisher",
        "story-first evaluator",
        "audience translator",
        "calm contrarian",
        "systems thinker",
    ]
    risk_map = {
        "safe": "low",
        "balanced": "medium",
        "experimental": "high",
    }

    # Professional profiles (50)
    pro_combos = list(
        product(
            primary_roles,
            secondary_roles,
            ITEM_TYPES,
            AUDIENCES,
            TONES,
            NOVELTY_LEVELS,
            EVIDENCE_LEVELS,
            CTA_STYLES,
            PACE_LEVELS,
            VISUAL_STYLES,
            VALUES,
        )
    )
    rng.shuffle(pro_combos)

    # Hater profiles (50) - selfish/negative, VERY different from each other
    hater_combos = list(
        product(
            HATER_PRIMARY_ROLES,
            HATER_SECONDARY_ROLES,
            ITEM_TYPES,
            AUDIENCES,
            TONES,
            NOVELTY_LEVELS,
            EVIDENCE_LEVELS,
            CTA_STYLES,
            PACE_LEVELS,
            VISUAL_STYLES,
            VALUES,
        )
    )
    rng.shuffle(hater_combos)

    # 50 professional + 50 hater, interleaved
    n_pro, n_hater = n // 2, n - n // 2
    combos = list(pro_combos[:n_pro]) + list(hater_combos[:n_hater])
    rng.shuffle(combos)

    profiles: list[HybridProfile] = []
    used_descriptions: set[str] = set()
    for idx, combo in enumerate(combos):
        (
            primary_role,
            secondary_role,
            item_type,
            audience,
            tone,
            novelty,
            evidence,
            cta_style,
            pace,
            visual_style,
            value,
        ) = combo
        is_hater = primary_role in HATER_PRIMARY_ROLES
        name = f"{primary_role[:30]} / {secondary_role[:30]} #{idx + 1}"
        if is_hater:
            description = (
                f"A negative, selfish reviewer who is a {primary_role} and a {secondary_role}. "
                f"They tend to tear down work, find faults, and rarely engage constructively. "
                f"They are most interested in {item_type}s aimed at {audience}. "
                f"They prefer a {tone} tone, {novelty} ideas, {evidence} evidence, and a {cta_style} call to action. "
                f"They respond best to a {pace} pacing style, a {visual_style} presentation, and work that prioritizes {value}. "
                f"They have {risk_map[novelty]} tolerance for creative risk."
            )
        else:
            description = (
                f"A hybrid reviewer who thinks like a {primary_role} and a {secondary_role}. "
                f"They are most interested in {item_type}s aimed at {audience}. "
                f"They prefer a {tone} tone, {novelty} ideas, {evidence} evidence, and a {cta_style} call to action. "
                f"They respond best to a {pace} pacing style, a {visual_style} presentation, and work that prioritizes {value}. "
                f"They have {risk_map[novelty]} tolerance for creative risk."
            )
        if description in used_descriptions:
            continue
        used_descriptions.add(description)
        profiles.append(
            HybridProfile(
                name=name,
                description=description,
                preferred_item_type=item_type,
                audience=audience,
                tone=tone,
                novelty=novelty,
                evidence=evidence,
                cta_style=cta_style,
                pace=pace,
                visual_style=visual_style,
                value=value,
                risk_tolerance=risk_map[novelty],
            )
        )
        if len(profiles) == n:
            return profiles

    raise RuntimeError(f"Could not generate {n} unique hybrid profiles")


def build_question_pool(n: int, seed: int) -> list[ReviewPrompt]:
    rng = random.Random(seed)
    subject_lines = [
        "It opens with a direct promise and closes with a single ask.",
        "It spends most of its space explaining the value proposition.",
        "It uses a first-person story before making the pitch.",
        "It leans on proof points and short customer quotes.",
        "It is built around a fast hook and a strong last line.",
        "It frames the idea as a practical fix for a common headache.",
        "It tries to sound premium without using much detail.",
        "It uses a playful metaphor to make the idea memorable.",
        "It focuses on urgency and a tight deadline.",
        "It slows down to explain tradeoffs and next steps.",
    ]
    benefit_lines = [
        "The core promise is better coordination with less effort.",
        "The main benefit is faster feedback for the target audience.",
        "The pitch centers on trust, reliability, and fewer mistakes.",
        "The concept highlights convenience and immediate payoff.",
        "The message positions the offer as thoughtful and human.",
        "The narrative emphasizes momentum and visible progress.",
        "The copy frames the work as bold, modern, and hard to ignore.",
        "The paragraph suggests the team has done real homework.",
        "The creative tries to feel calm, useful, and easy to share.",
        "The piece wants the audience to act now, not later.",
    ]
    weakness_lines = [
        "Some lines feel generic and a little overpolished.",
        "A few claims arrive without enough proof.",
        "The idea is clear, but the finish is slightly flat.",
        "The tone is confident, though it risks sounding rehearsed.",
        "The structure is neat, but it may be too safe.",
        "The paragraph is vivid, yet parts of it feel crowded.",
        "The creative is energetic, but it could overwhelm some viewers.",
        "The draft is credible, although the close is understated.",
        "The campaign has heart, but the signal can get noisy.",
        "The concept is strong, but the ask may be a bit aggressive.",
    ]

    combos = list(
        product(
            ITEM_TYPES,
            AUDIENCES,
            TONES,
            NOVELTY_LEVELS,
            EVIDENCE_LEVELS,
            CTA_STYLES,
            PACE_LEVELS,
            VISUAL_STYLES,
            VALUES,
        )
    )
    rng.shuffle(combos)

    prompts: list[ReviewPrompt] = []
    used_questions: set[str] = set()
    for idx, combo in enumerate(combos):
        (
            item_type,
            audience,
            tone,
            novelty,
            evidence,
            cta_style,
            pace,
            visual_style,
            value,
        ) = combo
        subject = subject_lines[idx % len(subject_lines)]
        benefit = benefit_lines[(idx * 3) % len(benefit_lines)]
        weakness = weakness_lines[(idx * 7) % len(weakness_lines)]
        paragraph = (
            f"This {item_type} targets {audience} with a {tone} tone. "
            f"It takes a {novelty} approach, uses {evidence} supporting detail, and has a {cta_style} call to action. "
            f"The pacing feels {pace} and the presentation reads as {visual_style}. "
            f"It is trying to maximize {value}. {subject} {benefit} {weakness}"
        )
        prompt = ReviewPrompt(
            question_id=idx,
            item_type=item_type,
            audience=audience,
            tone=tone,
            novelty=novelty,
            evidence=evidence,
            cta_style=cta_style,
            pace=pace,
            visual_style=visual_style,
            value=value,
            paragraph=paragraph,
        )
        rendered = prompt.render()
        if rendered in used_questions:
            continue
        used_questions.add(rendered)
        prompts.append(prompt)
        if len(prompts) == n:
            return prompts

    raise RuntimeError(f"Could not generate {n} unique review prompts")


def preference_score(profile: HybridProfile, prompt: ReviewPrompt) -> int:
    score = 0
    if profile.preferred_item_type == prompt.item_type:
        score += 2
    if profile.audience == prompt.audience:
        score += 2
    if profile.tone == prompt.tone:
        score += 1
    if profile.novelty == prompt.novelty:
        score += 2
    elif {profile.novelty, prompt.novelty} == {"safe", "experimental"}:
        score -= 2
    if profile.evidence == prompt.evidence:
        score += 2
    elif {profile.evidence, prompt.evidence} == {"light", "heavy"}:
        score -= 2
    if profile.cta_style == prompt.cta_style:
        score += 1
    elif {profile.cta_style, prompt.cta_style} == {"none", "hard"}:
        score -= 1
    if profile.pace == prompt.pace:
        score += 1
    if profile.visual_style == prompt.visual_style:
        score += 1
    if profile.value == prompt.value:
        score += 2
    return score


def score_to_label(score: int) -> str:
    if score >= 9:
        return "strong_like"
    if score >= 5:
        return "like"
    if score >= 2:
        return "neutral"
    if score >= -1:
        return "dislike"
    return "strong_dislike"


def candidate_answers(profile: HybridProfile, prompt: ReviewPrompt) -> dict[str, list[str]]:
    positive_reason = (
        f"the {prompt.tone} tone fits the audience, the {prompt.evidence} evidence level feels right, "
        f"and the focus on {prompt.value} comes through clearly"
    )
    mismatch_reason = (
        f"the {prompt.tone} tone misses the audience, the {prompt.evidence} evidence level feels off, "
        f"and the push toward {prompt.value} is not convincing enough"
    )
    return {
        "strong_like": [
            f"I absolutely love this {prompt.item_type}. {positive_reason.capitalize()}.",
            f"I strongly like this {prompt.item_type}; it feels sharp, effective, and genuinely compelling for {prompt.audience}.",
            f"I really love this piece. It is confident, well aimed, and the execution feels excellent.",
        ],
        "like": [
            f"I like this {prompt.item_type}. {positive_reason.capitalize()}.",
            f"I like it overall; the idea is solid and the execution feels pretty effective for {prompt.audience}.",
            f"This works for me. It is appealing, clear, and mostly well judged.",
        ],
        "neutral": [
            f"I feel neutral about this {prompt.item_type}. It is competent, but it does not strongly move me either way.",
            f"My reaction is neutral. I can see what it is trying to do, but it lands in a fairly average way.",
            f"I am neutral on this one. Parts of it work, and parts of it feel ordinary.",
        ],
        "dislike": [
            f"I do not like this {prompt.item_type}. {mismatch_reason.capitalize()}.",
            f"I dislike it overall; the idea feels weak and the execution is not very convincing.",
            f"This does not work for me. It feels off-target and not especially effective.",
        ],
        "strong_dislike": [
            f"I really dislike this {prompt.item_type}. {mismatch_reason.capitalize()}.",
            f"I strongly dislike it; the whole thing feels misguided, unconvincing, and frustrating to read.",
            f"I hate this direction. It feels badly matched to {prompt.audience} and the execution is poor.",
        ],
    }


def select_answer_for_label(
    analyzer: SentimentIntensityAnalyzer,
    profile: HybridProfile,
    prompt: ReviewPrompt,
    target_label: str,
    rng: random.Random,
) -> tuple[str, dict[str, float]]:
    options = candidate_answers(profile, prompt)[target_label]
    rng.shuffle(options)
    for option in options:
        scores = analyzer.polarity_scores(option)
        if vader_label(scores["compound"]) == target_label:
            return option, scores
    raise RuntimeError(f"No valid answer found for label={target_label}")


def build_dataset(seed: int = 42) -> dict:
    rng = random.Random(seed)
    analyzer = SentimentIntensityAnalyzer()
    profiles = build_hybrid_profiles(NUM_DESCRIPTIONS, seed)
    question_pool = build_question_pool(QUESTION_POOL_SIZE, seed + 1)

    responses = []
    for profile_idx, profile in enumerate(profiles):
        chosen_questions = rng.sample(question_pool, k=QUESTIONS_PER_DESCRIPTION)
        qa_pairs = []
        for prompt in chosen_questions:
            score = preference_score(profile, prompt)
            label = score_to_label(score)
            answer, scores = select_answer_for_label(analyzer, profile, prompt, label, rng)
            qa_pairs.append(
                {
                    "question_id": prompt.question_id,
                    "question": prompt.render(),
                    "answer": answer,
                    "sentiment_label": label,
                    "compatibility_score": score,
                    "vader": scores,
                }
            )

        responses.append(
            {
                "id": profile_idx,
                "hybrid_name": profile.name,
                "description": profile.description,
                "preferences": {
                    "preferred_item_type": profile.preferred_item_type,
                    "audience": profile.audience,
                    "tone": profile.tone,
                    "novelty": profile.novelty,
                    "evidence": profile.evidence,
                    "cta_style": profile.cta_style,
                    "pace": profile.pace,
                    "visual_style": profile.visual_style,
                    "value": profile.value,
                    "risk_tolerance": profile.risk_tolerance,
                },
                "qa_pairs": qa_pairs,
            }
        )

    return {
        "description_count": NUM_DESCRIPTIONS,
        "question_pool_size": QUESTION_POOL_SIZE,
        "questions_per_description": QUESTIONS_PER_DESCRIPTION,
        "question_pool": [
            {
                "question_id": prompt.question_id,
                "item_type": prompt.item_type,
                "audience": prompt.audience,
                "tone": prompt.tone,
                "novelty": prompt.novelty,
                "evidence": prompt.evidence,
                "cta_style": prompt.cta_style,
                "pace": prompt.pace,
                "visual_style": prompt.visual_style,
                "value": prompt.value,
                "question": prompt.render(),
            }
            for prompt in question_pool
        ],
        "responses": responses,
    }


def write_dataset(output_path: str | Path, seed: int = 42) -> Path:
    output_path = Path(output_path)
    dataset = build_dataset(seed=seed)
    with output_path.open("w") as f:
        json.dump(dataset, f, indent=2)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hybrid review dataset")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = write_dataset(args.output, seed=args.seed)
    print(
        f"Wrote {NUM_DESCRIPTIONS} hybrid descriptions, "
        f"{QUESTION_POOL_SIZE} unique questions, "
        f"and {NUM_DESCRIPTIONS * QUESTIONS_PER_DESCRIPTION} QA pairs -> {path}"
    )
