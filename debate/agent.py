import ollama
from prompts import STANCES, PERSONALITIES, ROLES, GUIDELINES

AGENT_MODEL = "llama3.2"

class DebateAgent:
    def __init__(self, stance: str, personality: str):
        self.stance = stance
        self.personality = personality
        self.conversation_history = []

    def _build_system_prompt(self) -> str:
        return f"""{STANCES[self.stance]}

{PERSONALITIES[self.personality]}

{GUIDELINES}"""

    def generate(self, role: str, opponent_last_turn: str = None) -> str:
        system_prompt = self._build_system_prompt()
        role_prompt = ROLES[role]

        if opponent_last_turn:
            user_message = f"""{role_prompt}

Your opponent just said:
\"\"\"{opponent_last_turn}\"\"\"

Now deliver your response."""
        else:
            user_message = f"""{role_prompt}

Now deliver your opening statement."""

        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        response = ollama.chat(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                *self.conversation_history
            ]
        )

        reply = response["message"]["content"]

        self.conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        return reply

    def reset(self):
        self.conversation_history = []
