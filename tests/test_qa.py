import json
from pathlib import Path

from services.qa_service import QAService

analysis = json.loads(Path("outputs/audio2_analysis.json").read_text())
qa = QAService()

questions = [
    "What was said?",
    "Which language was spoken?",
    "Was anyone speaking?",
    "What background sounds were there?",
    "Was there a dog?",
    "Was there a cat?",
    "What sounds were heard when the speaker said good idea?",
    "What was happening at 10 seconds?",
    "Summarize the recording",
    "How reliable is this?",
    "What colour was the car?",
]

for q in questions:
    r = qa.answer(q, analysis)
    print(f"\nQ: {q}\nA [{r['answer_type']}]: {r['answer']}")

assert qa.answer("What colour was the car?", analysis)["grounded_in_analysis"] is False
assert qa.answer("What was said?", analysis)["answer_type"] == "observed"
assert qa.answer("Was there a dog?", analysis)["answer_type"] == "insufficient"
print("\nPASS: QA")