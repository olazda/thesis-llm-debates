import time
from debate.runner import run_debate
from debate.config import MOTIONS, PERSONALITIES_LIST

def run_all_debates():
    total = len(PERSONALITIES_LIST) * len(MOTIONS)
    count = 0

    print(f"Starting {total} debates...\n")

    for personality in PERSONALITIES_LIST:
        for motion in MOTIONS:
            count += 1
            print(f"\n[{count}/{total}] Running debate...")
            try:
                run_debate(
                    stance_a="pro_immigration",
                    personality_a=personality,
                    stance_b="restrictive_immigration",
                    personality_b=personality,
                    motion=motion
                )
                time.sleep(2)
            except Exception as e:
                print(f"ERROR in debate {count}: {e}")
                print("Skipping and continuing...\n")
                continue

    print(f"\nAll {total} debates complete.")
    print(f"Results saved to storage/debates/")

if __name__ == "__main__":
    run_all_debates()