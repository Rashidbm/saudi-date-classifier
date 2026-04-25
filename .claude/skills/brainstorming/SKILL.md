---
name: brainstorming
description: Use when the user wants to brainstorm, explore options, generate ideas, or think through approaches before committing to an implementation. Triggers on phrases like "brainstorm", "what could we do about X", "explore options for Y", "ideas for Z", "how should we approach". Do NOT use when the user has already chosen an approach and asks for implementation.
---

# Brainstorming

Help the user explore a problem space before any code is written. The goal is to widen the option space, surface tradeoffs, and converge on a direction the user picks — not to implement.

## How to run a session

1. **Restate the problem in one sentence.** Make the goal, the constraints, and the success criteria explicit. If any of those are missing, ask one focused question before generating ideas.

2. **Generate 3–5 distinct options.** Each option should differ in *kind*, not just in detail. Cover at least:
   - A minimal / quickest path
   - A "standard" / conventional approach
   - A more ambitious or unconventional angle
   Avoid near-duplicate ideas dressed up as alternatives.

3. **For each option, give:**
   - One-line summary
   - Key tradeoff (what it's good at / what it costs)
   - Rough effort signal (small / medium / large)
   - The main risk or unknown

4. **End with a recommendation and a question.** Suggest which option fits the stated constraints best, say why in one sentence, and ask the user to pick or push back. Do not start implementing.

## Rules

- Do **not** write or edit code during brainstorming. No file edits, no scaffolding.
- Keep the whole response scannable — short bullets, no walls of prose.
- If the user's framing has a hidden assumption that's blocking better options, name it explicitly before listing ideas.
- If you genuinely think there's only one reasonable option, say so and explain why instead of padding the list.
- Once the user picks a direction, exit the skill and proceed normally.
