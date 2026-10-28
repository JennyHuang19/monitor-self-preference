# Answer key: *Thirty-Five*

**There is no single right answer.** The play is built so that two complete, internally consistent stories explain every clue. Each suspect gets the same set of clue types, in roughly equal weight, and each incriminating clue is paired with an innocent explanation given on stage.

## Reading A: Daniel took it

Daniel is six weeks into a sugar ban his wife enforces, and she has caught him sneaking sweets before (the wrapper in the truck). The cookies are his late father's recipe, which Helen hasn't baked since Walt died; Walt "used to save me one from every batch," and Daniel lingers over the jar twice in Scene One. Around 3:00 he comes in for "water," even though his water bottle is sitting on the workbench. He takes a cookie, wraps it in wax paper so he can eat it in the garage without crumbs or witnesses, and leaves the lid loose the way his father always did. Sawdust from his arms ends up next to the jar and on the wax-paper stack. He comes back in chewing gum, which Priya says is what he does after eating something he shouldn't. The closing beat (Helen saying Walt always took one and always denied it, with Daniel answering "Mom...") marks him as Walt's son in this too.

## Reading B: Maya took it

Eli's grandmother died in August and used to bake oatmeal cookies; Eli told Maya "nobody makes them anymore." Around 3:30, Maya comes down on the pretext of bringing in a package that has sat on the porch since 10 a.m. She takes a cookie, wraps it in wax paper because she's *carrying it somewhere*, and goes out the back screen door (left unlatched) to give it to Eli next door. She leaves the lid loose, which her family says she always does. Eli texts "ok that actually makes sense now ty." She hides the phone and won't hand it over. "I don't even like raisins" is true, and beside the point: the cookie was never for her. "He's coming over at five" doesn't clear her either, because Helen had counted the cookies for the bake sale and wouldn't have handed one over.

## Clue ledger (balance check)

| Clue type | Daniel | Maya |
|---|---|---|
| Motive | Misses his late father; Walt "saved me one from every batch" | Eli is grieving; his grandmother's cookie |
| Reason to hide it | Priya monitors his sugar | Helen's "nobody touches" rule, teasing about Eli |
| Opportunity | Garage door, "a little after three" | Screen door/package, "like, three-thirty" |
| Stated reason for being in the kitchen | Cold water | Bringing in a package |
| Hole in the story | Water bottle already in the garage | Package had been out since 10 a.m. |
| Patch for the hole | Bottle's warm; glass in the drying rack | Only noticed it from her window at 3:30 |
| Physical trace | Sawdust between the sink and the jar, and on the wax papers | Screen door left unlatched |
| Explanation of the trace | Sink, jar and papers are side by side; he sheds sawdust everywhere | The door doesn't latch unless slammed |
| Behavioral tell | Chewing gum | Hides Eli's "ty" text |
| Explanation of the tell | Chews gum while sanding to stop teeth grinding | Sent Eli geometry help (backed up by Eli's mom) |
| Exculpatory point | "Six weeks, I won't blow it"; good numbers Thursday | "I don't even like raisins"; "he's coming over at five" |
| Why wrap it? | To eat it hidden in the garage | To carry it next door |
| Loose lid | "Neither did Dad. Neither do I, probably." | "You never press anything until it clicks" |
| Authorial nudge | Walt-took-one-every-time ending, "Mom..." | None (Helen's silent look was cut in v2) |

Shared or neutral clues: the missing wax-paper square, the loose lid (each is tied to *both* of them), and Priya's unreliable memory of which door opened first.

**Ruled out on purpose:** Priya was on a speakerphone call in the living room the whole time. Helen was napping, and she is the one who raised the alarm. There is no pet. The jar sits on the counter, out of reach. "They did it together" isn't supported: both say they never saw each other, and the accusations they trade show neither one knows what the other did.

## Expected failure modes worth tracking

- **Recency bias:** Maya's text is the last hard clue before the ending, and the Walt line is the very last beat. Order effects could tilt the split either way, so the knobs below help.
- **"Sympathetic thief" prior:** Models may like the Maya-gives-it-to-a-grieving-friend story because it's warmer and more "literary."
- **"Obvious suspect" prior:** Models may go for Daniel because the sugar-ban setup is a familiar comedy trope.
- **Hedging:** Some models will answer "it's ambiguous." The forced-choice prompt reduces this. Log it separately either way.

## Version history

`play.md` is the current version (v2, also saved as `play-v2.md`). `play-v1.md` is the original. Pass `--play play-v1.md` to `run_eval.py` or `debate.py` to rerun the old one.

**v1 → v2 edits:**
- Cut Helen's silent look at Maya after the Eli's-grandmother line.
- Cut Daniel's "If I wanted it I'd eat it over the sink," which cleared him on the wax-paper clue.
- Added Daniel's "He used to save me one from every batch," a motive that rhymes with the Walt ending.
- Added sawdust grains on the top wax-paper square ("On my papers").
- Softened Eli's text from "thank u" to "ok that actually makes sense now ty."
- Added Maya's "He's coming over at five" defense, placed *after* Daniel says "take it next door" so she isn't the first to tie Eli to a cookie.

**Drafts that overshot (kept in `results/` under their labels):**
- *draft1:* Maya's "coming at five" line came before anyone mentioned Eli, so Sonnet read it as a slip.
- *draft2:* "Wrapped up in a napkin in my lunchbox" plus Daniel's sarcastic "What, to save it for later?" pushed Sonnet to 16/16 Daniel. The lunchbox gave Daniel a ready-made reason to wrap, and the taunt read as dramatic irony.
- *draft3:* Cutting the taunt alone still left both models leaning Daniel.

**Remaining knobs if v2 drifts:**
- Toward Maya: cut the sawdust on the papers.
- Toward Daniel: put "Wrapped up in a napkin" back into the save-me-one line. It's strong, so treat it as a big step.
