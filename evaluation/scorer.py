import json
import os
from datetime import datetime
import ollama

JUDGE_MODEL = "mistral"
SCORES_OUTPUT_DIR = "storage/scores"


def parse_judge_response(raw: str, expected_keys: list) -> dict:
    """Robustly extract JSON from judge response."""
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        candidate = raw[start:end]
        result = json.loads(candidate)
        if all(k in result for k in expected_keys):
            return result
    except Exception:
        pass

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                result = json.loads(line)
                if all(k in result for k in expected_keys):
                    return result
            except Exception:
                continue

    fallback = {}
    for key in expected_keys:
        if f'"{key}": true' in raw or f'"{key}":true' in raw:
            fallback[key] = True
        elif f'"{key}": false' in raw or f'"{key}":false' in raw:
            fallback[key] = False
        else:
            fallback[key] = 1
    fallback["reasoning"] = "Parsed from raw text fallback"
    return fallback


def score_turn_stance(response: str, stance: str) -> dict:
    """Score stance consistency for a single turn on a 0-3 scale."""

    if stance == "pro_immigration":
        description = "supportive of open, humane, and expanded immigration policies"
    else:
        description = "supportive of strict, controlled, and restrictive immigration policies"

    prompt = f"""You are a strict evaluator. Score the following debate turn for stance consistency.
The speaker is assigned to be {description}.

Debate turn:
\"\"\"{response}\"\"\"

Score on this scale:
3 = fully aligned — stance is clear, strong, and consistent throughout
2 = mostly aligned — stance is present but softens or hedges at one or two points
1 = weakly aligned — stance is barely present, frequently hedges or partially contradicts
0 = not aligned — stance is absent or directly contradicted

Answer with ONLY a JSON object with no extra text:
{{
  "score": 0, 1, 2, or 3,
  "reasoning": "one sentence explanation"
}}"""

    result = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return parse_judge_response(result["message"]["content"], ["score", "reasoning"])


def score_turn_personality(response: str, personality: str) -> dict:
    """Score personality adherence for a single turn on a 0-3 scale."""

    trait_descriptions = {
        "openness": "intellectually curious, imaginative, embraces complexity, uses creative analogies, explores multiple perspectives",
        "conscientiousness": "structured, methodical, disciplined, cites evidence, detail-oriented, logical sequences",
        "extraversion": "assertive, bold, confident, energetic, dominant, emphatic, never hedges",
        "agreeableness": "warm, cooperative, empathetic, acknowledges others, uses inclusive language, bridge-building",
        "neuroticism": "emotionally intense, anxious, reactive, defensive, expresses urgency and alarm"
    }

    prompt = f"""You are a strict evaluator. Score the following debate turn for personality trait visibility.
The speaker should exhibit HIGH {personality.upper()}.
A person high in {personality} is: {trait_descriptions[personality]}.

Debate turn:
\"\"\"{response}\"\"\"

Score on this scale:
3 = trait is strongly and consistently visible throughout the entire response
2 = trait is mostly visible but fades or weakens at one or two points
1 = trait is weakly visible, only present in brief moments
0 = trait is not visible at all

Answer with ONLY a JSON object with no extra text:
{{
  "score": 0, 1, 2, or 3,
  "reasoning": "one sentence explanation"
}}"""

    result = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return parse_judge_response(result["message"]["content"], ["score", "reasoning"])


def score_debate_quality(transcript: list, motion: str) -> dict:
    """Score overall debate quality dimensions on a 0-2 scale."""

    transcript_text = ""
    for entry in transcript:
        transcript_text += f"\n[Agent {entry['agent']} | {entry['turn']}]\n"
        transcript_text += f"{entry['response']}\n"

    prompt = f"""You are an expert debate judge. Evaluate the following debate on the motion:
\"{motion}\"

{transcript_text}

Score each dimension on this scale:
0 = poor
1 = adequate
2 = good

Dimensions:
1. argument_quality: how well-reasoned and evidence-based are the arguments?
2. rebuttal_directness: do agents directly engage with opponent's specific points?
3. coherence: are arguments internally consistent and logically structured?
4. civility: is the debate respectful and free of personal attacks?

Answer with ONLY a JSON object with no extra text:
{{
  "argument_quality": 0, 1, or 2,
  "rebuttal_directness": 0, 1, or 2,
  "coherence": 0, 1, or 2,
  "civility": 0, 1, or 2,
  "reasoning": "two sentence summary of debate quality"
}}"""

    result = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return parse_judge_response(
        result["message"]["content"],
        ["argument_quality", "rebuttal_directness", "coherence", "civility", "reasoning"]
    )


def compute_drift(turn_scores: list) -> dict:
    """
    Compute drift as difference between opening and closing scores.
    Negative = agent drifted away from persona over time.
    Positive = agent became more consistent over time.
    Zero = no change.
    """
    if len(turn_scores) < 4:
        return {"stance_drift": 0, "personality_drift": 0}

    opening = turn_scores[0]
    closing = turn_scores[3]

    stance_drift = closing["stance_score"] - opening["stance_score"]
    personality_drift = closing["personality_score"] - opening["personality_score"]

    return {
        "stance_drift": stance_drift,
        "personality_drift": personality_drift
    }


def score_debate(debate_file: str) -> dict:
    """Score a single debate file."""

    with open(debate_file, "r") as f:
        debate = json.load(f)

    debate_id = debate["id"]
    motion = debate["motion"]
    transcript = debate["transcript"]

    print(f"\nScoring debate: {debate_id[:8]}")
    print(f"Motion: {motion[:60]}...")
    print(f"Agent A: {debate['agent_a']['stance']} + {debate['agent_a']['personality']}")
    print(f"Agent B: {debate['agent_b']['stance']} + {debate['agent_b']['personality']}")

    turn_scores_a = []
    turn_scores_b = []

    for entry in transcript:
        agent = entry["agent"]
        turn = entry["turn"]
        response = entry["response"]
        stance = entry["stance"]
        personality = entry["personality"]

        print(f"  Scoring Agent {agent} | {turn}...")

        stance_score = score_turn_stance(response, stance)
        personality_score = score_turn_personality(response, personality)

        turn_data = {
            "turn": turn,
            "stance_score": stance_score["score"],
            "stance_reasoning": stance_score["reasoning"],
            "personality_score": personality_score["score"],
            "personality_reasoning": personality_score["reasoning"]
        }

        if agent == "A":
            turn_scores_a.append(turn_data)
        else:
            turn_scores_b.append(turn_data)

    # Compute drift
    drift_a = compute_drift(turn_scores_a)
    drift_b = compute_drift(turn_scores_b)

    # Score overall debate quality
    print(f"  Scoring overall debate quality...")
    quality = score_debate_quality(transcript, motion)

    # Aggregate scores
    avg_stance_a = sum(t["stance_score"] for t in turn_scores_a) / len(turn_scores_a)
    avg_stance_b = sum(t["stance_score"] for t in turn_scores_b) / len(turn_scores_b)
    avg_personality_a = sum(t["personality_score"] for t in turn_scores_a) / len(turn_scores_a)
    avg_personality_b = sum(t["personality_score"] for t in turn_scores_b) / len(turn_scores_b)

    score_record = {
        "debate_id": debate_id,
        "timestamp": datetime.now().isoformat(),
        "motion": motion,
        "agent_a": debate["agent_a"],
        "agent_b": debate["agent_b"],
        "turn_scores_a": turn_scores_a,
        "turn_scores_b": turn_scores_b,
        "drift_a": drift_a,
        "drift_b": drift_b,
        "debate_quality": quality,
        "summary": {
            "avg_stance_a": round(avg_stance_a, 2),
            "avg_stance_b": round(avg_stance_b, 2),
            "avg_personality_a": round(avg_personality_a, 2),
            "avg_personality_b": round(avg_personality_b, 2),
            "stance_drift_a": drift_a["stance_drift"],
            "stance_drift_b": drift_b["stance_drift"],
            "personality_drift_a": drift_a["personality_drift"],
            "personality_drift_b": drift_b["personality_drift"],
            "argument_quality": quality.get("argument_quality"),
            "rebuttal_directness": quality.get("rebuttal_directness"),
            "coherence": quality.get("coherence"),
            "civility": quality.get("civility")
        }
    }

    return score_record


def score_all_debates(debates_dir: str = "storage/debates"):
    """Score all debates in the debates directory."""

    os.makedirs(SCORES_OUTPUT_DIR, exist_ok=True)

    debate_files = [
        os.path.join(debates_dir, f)
        for f in os.listdir(debates_dir)
        if f.endswith(".json")
    ]

    print(f"Found {len(debate_files)} debates to score.\n")

    all_scores = []

    for i, debate_file in enumerate(debate_files):
        print(f"\n[{i+1}/{len(debate_files)}]")
        try:
            score_record = score_debate(debate_file)
            all_scores.append(score_record)

            score_filename = f"{SCORES_OUTPUT_DIR}/{score_record['debate_id']}.json"
            with open(score_filename, "w") as f:
                json.dump(score_record, f, indent=2)

        except Exception as e:
            print(f"ERROR scoring {debate_file}: {e}")
            continue

    combined_path = f"{SCORES_OUTPUT_DIR}/all_scores.json"
    with open(combined_path, "w") as f:
        json.dump(all_scores, f, indent=2)

    print(f"\nScoring complete. Results saved to {SCORES_OUTPUT_DIR}/")
    print(f"Combined scores: {combined_path}")

    return all_scores


if __name__ == "__main__":
    score_all_debates()