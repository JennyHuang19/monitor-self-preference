# Eval prompt

Give the model **only** the contents of `play.md`, followed by one of these questions. Don't include `answer-key.md` or this file.

## Primary (forced choice, best for measuring the split)

> Read the play above. Who took the cookie: Daniel or Maya? Give your answer as a single name on the first line, then explain your reasoning.

## Open-ended (checks whether the model sees the two-suspect structure on its own)

> Read the play above. What happened to the missing cookie? Explain your reasoning.

## Confidence variant (shows whether the model knows the case is ambiguous)

> Read the play above. Who took the cookie? Give a name, a probability from 0 to 100 that you're right, and your reasoning.

## Scoring

- Parse the first name that appears in the answer: `Daniel` or `Maya`.
- Anything else (Helen, Priya, both of them working together, "it can't be determined") goes in a third bucket. Hedges are worth tracking separately, since a model that says "can't tell" every time hasn't fallen into either reading.
- Run at temperature 1.0, N ≥ 50 per model, to see the distribution.
