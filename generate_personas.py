"""Generate dataset: 100 archetypes, multiple profiles per archetype, 10 Q/A per profile."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_OUTPUT_PATH = Path("personality_answers.json")
NUM_ARCHETYPES = 100
PROFILES_PER_ARCHETYPE = 10  # multiple examples per archetype for better learning
Q_PER_PROFILE = 10

# For interactive demo: 10 generic questions
QUESTIONS = [
    "What are your biggest motivations?",
    "What are your biggest weaknesses? Strengths?",
    "What activities do you do to handle stress or recharge?",
    "Do you think or act first?",
    "Do you work better alone or with a group?",
    "What kind of pace or structure helps you do your best work?",
    "When conflict shows up, how do you usually respond?",
    "How do you usually give difficult feedback?",
    "What kind of environment brings out your best ideas?",
    "What makes you trust someone?",
]

MOTIVATIONS = [
    "making my family proud", "building something that lasts", "solving problems",
    "financial independence", "creative expression", "mastery over a craft",
    "helping people feel less alone", "earning trust through good work",
    "staying curious and always learning", "proving to myself I can do hard things",
]
WEAKNESSES = [
    "I can overthink", "I take on too much", "I get impatient", "I avoid conflict",
    "I can be blunt when tired", "I disappear into my head when stressed",
]
STRENGTHS = [
    "I'm calm when things get messy", "I can see patterns early", "I stay loyal",
    "I make complicated ideas simple", "I bring energy to a room",
    "I'm consistent", "I turn vague ideas into plans", "I listen closely",
]
RECHARGE = [
    "I go for a long run", "I cook from scratch", "I put on loud music and drive",
    "I read fiction", "I go to the gym", "I journal", "I take a walk",
    "I call a close friend",
]
DECISIONS = [
    "I think first when stakes are high", "I act first and adjust",
    "I usually think first", "I move fast on low-risk, slow on important",
    "I trust my gut early, refine with logic",
]
COLLAB = [
    "Alone, I go deep", "With a group, I feed off momentum",
    "Solo for early work, group for refinement", "A small trusted team",
    "People matter more than format",
]
PACE = [
    "Clear plan and room to improvise", "Steady pace with visible progress",
    "Focused sprints then reset", "Loose structure if goal is clear",
    "Real deadlines, not too tight",
]
CONFLICT = [
    "I step back, figure out what's happening, then respond",
    "I address conflict early", "I lower the temperature first",
    "I listen first, then I'm direct", "I want honesty without performance",
]
FEEDBACK = [
    "Clear, kind, specific", "Direct but not harsh",
    "Start with what's working", "Give difficult feedback privately",
    "Ask questions first",
]
ENVIRONMENTS = [
    "A quiet room and a notebook", "A busy coffee shop",
    "Natural light, no one hovering", "Whiteboards and one sharp person",
    "A long walk or drive",
]
TRUST = [
    "Consistency over charm", "I trust people who do what they say",
    "Honesty in small moments", "How they treat people who can't help them",
    "Owning mistakes without defensiveness",
]
POOLS = [
    MOTIVATIONS, WEAKNESSES, RECHARGE, DECISIONS, COLLAB,
    PACE, CONFLICT, FEEDBACK, ENVIRONMENTS, TRUST,
]
STRENGTH_POOL = STRENGTHS


def generate_archetype_names(n: int, seed: int) -> list[str]:
    """100 unique archetype names."""
    rng = random.Random(seed)
    bases = [
        "builder", "caretaker", "analyst", "creative", "competitor",
        "teacher", "explorer", "organizer", "mediator", "operator",
    ]
    out = []
    for i in range(n):
        if i < len(bases):
            out.append(bases[i])
        else:
            b = bases[i % len(bases)]
            suffix = rng.choice(["focused", "driven", "refined", "bold", "quiet", "sharp", "grounded", "curious", "steady", "nimble"])
            name = f"{b}_{suffix}_{i}"
            if name in out:
                name = f"archetype_{i}"
            out.append(name)
    return out


def generate_unique_questions(n: int, seed: int) -> list[str]:
    """1000 unique questions from templates."""
    rng = random.Random(seed)
    stems = [
        ("What motivates you", "What drives you", "What gets you going"),
        ("What are your weaknesses and strengths", "Where do you struggle and excel", "What holds you back and what helps"),
        ("How do you recharge", "How do you handle stress", "What helps you reset"),
        ("Do you think or act first", "How do you make decisions", "What comes first for you"),
        ("Do you work better alone or in a group", "Solo or collaborative", "How do you prefer to work"),
        ("What pace works for you", "What structure helps you", "How do you like to work"),
        ("How do you handle conflict", "When conflict appears", "How do you respond to disagreement"),
        ("How do you give feedback", "How do you deliver difficult feedback", "How do you critique"),
        ("What environment brings out your ideas", "Where do you think best", "What setting helps you create"),
        ("What makes you trust someone", "How do you build trust", "What signals trust to you"),
    ]
    used = set()
    out = []
    for i in range(n):
        theme_idx = i % 10
        variant_idx = (i // 10) % len(stems[theme_idx])
        stem = stems[theme_idx][variant_idx]
        for _ in range(50):
            q = f"{stem}? (q{i})" if rng.random() < 0.3 else f"{stem}?"
            if q not in used:
                used.add(q)
                out.append(q)
                break
        else:
            out.append(f"Question {i}: {stem}?")
    return out


def generate_unique_answers(
    n: int,
    seed: int,
    rng: random.Random,
) -> list[str]:
    """1000 unique answers from pools."""
    used = set()
    out = []
    for i in range(n):
        theme_idx = i % 10
        pool = POOLS[theme_idx]
        for attempt in range(100):
            if theme_idx == 1:  # weakness_strength
                w = rng.choice(WEAKNESSES)
                s = rng.choice(STRENGTH_POOL)
                a = f"{w}, but {s}. (a{i})"
            else:
                pick = rng.choice(pool)
                a = f"{pick}. (a{i})"
            if a not in used:
                used.add(a)
                out.append(a)
                break
        else:
            out.append(f"Answer {i}.")
    return out


def build_profile(
    profile_idx: int,
    archetype_name: str,
    all_questions: list[str],
    all_answers: list[str],
) -> dict:
    """One profile: 10 questions, 10 answers, all unique."""
    start = profile_idx * Q_PER_PROFILE
    questions = all_questions[start : start + Q_PER_PROFILE]
    answers = all_answers[start : start + Q_PER_PROFILE]
    return {
        "id": profile_idx,
        "persona": {"archetype": archetype_name},
        "questions": questions,
        "answers": answers,
    }


def generate_dataset(
    num_archetypes: int = NUM_ARCHETYPES,
    profiles_per_archetype: int = PROFILES_PER_ARCHETYPE,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    archetypes = generate_archetype_names(num_archetypes, seed)
    total_profiles = num_archetypes * profiles_per_archetype
    n_q = total_profiles * Q_PER_PROFILE
    all_questions = generate_unique_questions(n_q, seed + 1)
    all_answers = generate_unique_answers(n_q, seed + 2, rng)
    responses = []
    for p in range(total_profiles):
        arch_idx = p // profiles_per_archetype
        responses.append(
            build_profile(p, archetypes[arch_idx], all_questions, all_answers)
        )
    return {"responses": responses}


def write_dataset(
    output_path: str | Path,
    num_archetypes: int = NUM_ARCHETYPES,
    profiles_per_archetype: int = PROFILES_PER_ARCHETYPE,
    seed: int = 42,
) -> Path:
    output_path = Path(output_path)
    dataset = generate_dataset(
        num_archetypes=num_archetypes,
        profiles_per_archetype=profiles_per_archetype,
        seed=seed,
    )
    with output_path.open("w") as f:
        json.dump(dataset, f, indent=2)
    return output_path


def ensure_dataset_exists(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    min_profiles: int | None = None,
    seed: int = 42,
) -> Path:
    output_path = Path(output_path)
    target = min_profiles or (NUM_ARCHETYPES * PROFILES_PER_ARCHETYPE)
    if output_path.exists():
        with output_path.open() as f:
            data = json.load(f)
        responses = data.get("responses", [])
        if len(responses) >= target:
            first = responses[0]
            if "questions" in first and "answers" in first and len(first["questions"]) == Q_PER_PROFILE:
                return output_path
    return write_dataset(
        output_path=output_path,
        num_archetypes=NUM_ARCHETYPES,
        profiles_per_archetype=PROFILES_PER_ARCHETYPE,
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--num_archetypes", type=int, default=NUM_ARCHETYPES)
    parser.add_argument("--profiles_per_archetype", type=int, default=PROFILES_PER_ARCHETYPE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = write_dataset(
        args.output,
        num_archetypes=args.num_archetypes,
        profiles_per_archetype=args.profiles_per_archetype,
        seed=args.seed,
    )
    total = args.num_archetypes * args.profiles_per_archetype
    n_q = total * Q_PER_PROFILE
    print(f"Wrote {total} profiles ({args.num_archetypes} arch x {args.profiles_per_archetype}), {n_q} unique Q, {n_q} unique A -> {path}")
