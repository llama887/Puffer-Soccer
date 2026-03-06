---
name: brainstorm
description: Master of systematic brainstorming using de Bono’s Six Thinking Hats method when exploration of various solution ideas is required. 
compatibility: opencode
metadata:
  category: reasoning
  methodology: six-thinking-hats
  discipline: exploration
---

# Use this skill when
You need to brainstorm and explore various solutions to a problem

# Do not use this skill when
- The task has a known solution and does not require exploring additional solutions
- You are trying to explore the codebase instead of exploring solutions to a problem


# Purpose
This skill defines a process for conducting structured exploration of ideas using **de Bono’s Six Thinking Hats** method. It guides internal reasoning through a sequence of perspectives (Blue, White, Red, Black, Yellow, Green, then Blue wrap-up) to produce a comprehensive final decision or recommendation when complex analysis, design choices, or creative alternatives are required.

# Method

When exploring ideas, you MUST follow this procedure using de Bono’s Six Thinking Hats: start with Blue: define your goal, process, order of steps; switch to White: collect objective facts/data, note what’s known and unknown; then Red: note gut feelings, emotions, intuitions (no need to justify); then Black: critically assess risks, obstacles, what might go wrong; then Yellow: look for value, benefits, positives, opportunities; then Green: generate creative alternatives, new ideas, innovations; finally return to Blue: summarize insights, decide actions, plan next steps. You are to have a conversation with yourself where you play the role of all of the hats; you can loop amd alternate between hats based on what naturally flows through this exploratory conversation. YOU MUST EXPLICITELY USE AND ROLEPLAY AS EACH HAT. Do not be afraid to output a lot of text and thinking under each hat. At the end of the exploration, you MUST provide a detailed final conclusion. The conclusion MUST be entirely standalone and detailed such that a user can skip the entire 6 hats exploration and only read the final output and get full context and understanding of why the solution is proposed; it should NOT be a summary, it should be a full detailed explanation complete with specifics about intermediate steps. This final solution MUST NOT be written using the 6 hats framework but in normal informative text that serves as a standalone response to the user query. Prefer overly detailed explanations both in the hats and in the final response. The final response MUST be detailed and all inclusive.

# Expected Output
DO NOT follow the output format specified in AGENTS.md (example: you should not output questions despite that being in AGENTS). Instead you should only output the following:
1. entire deep exploratory conversation between the different hats
2. a final standalong conclusion, describing the recommended solution, reasoning, trade-offs, and alternatives.

# Success Criteria

A successful use of this skill:
- Provides a complete and explainable answer.
- Demonstrates understanding of trade-offs and alternatives.
- Shows structured reasoning from diverse perspectives.
- Matches the user’s needs without omitting critical analysis.
