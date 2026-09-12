# Sample generations

All samples below are unedited output from the final checkpoint
(val loss 1.170), sampled with `temp=0.8`, `top_k=50`, `max_new=250`.
Prompts are shown in **bold** the text that comes after is entirely generated.

---

### Coherent narrative arc

**Once upon a time, a little boy named Camil** lived in a small house with his mom. Camil's mom was very strict and always said: "Camil, you must behave!"

One day, Camil wanted to play with his toys outside. As he went outside, he saw some police officers. Camil was a little scared, but he remembered to behave. He said hello to the police officers.

The police officers smiled and said: "Hello there! We see you have a good life."

Camil smiled and said: "Thank you! I'm just a little boy and I just wanted to play."

The police officers thanked Camil and said: "You are very polite. Please remember to always be careful and behave."

Camil nodded and said: "I will! Thank you for being so nice!"

---

### Dialogue and turn-taking

**He answered them**, "Yes, I want to!" His mom smiled and said, "Great! What do you want to do first?"

He thought for a few minutes, then shouted, "I want to play with my blocks!" His mom agreed and they went to get his blocks.

He was so proud when he put all the blocks together. He wanted to show everyone what he had built. His mom said, "Wow, you finished it so quickly! What do you want to do with it?"

He thought again, then he said, "Let's build something special!" He and his mom worked together to build a big castle.

---

### Emotional turn mid-story

**abdou was a little boy** who really enjoyed playing in the garden. One day, he decided to go and explore. He walked around, looking for something exciting.

Suddenly, he heard a noise from the bushes. He went a bit closer, and he saw a big, shiny box. He wondered what was inside.

He opened the box and found lots of toys and candy. He was so excited and he couldn't believe his eyes.

He looked around the garden, and then he saw something else. It was a big red balloon floating in the air. He wanted to get it.

But then the balloon suddenly floated away, and the little boy got very sad. He wanted to keep the balloon, but he knew he couldn't. So, he sadly walked back home with his empty cart.

---

### Short, fully-resolved story

**He was in the kitchen**, looking for something to eat. He opened the refrigerator and saw something yummy. He reached for it and saw that it was a tasty cookie. He took it out and put it on the plate. He ate it quickly and it had a sweet taste. He smiled. The cookie was so tasty! He looked for another one and found one. He opened it and ate it. It was so yummy. He kept looking for more tasty treats, and he was very happy. He was having a great time!

---

### Generalisation to unseen names

The model was never trained on these names, but byte-level BPE handles them
and they stay consistent through the story.

**Bouchra was a little girl** who loved the beach. Every day she went to the beach with her mom and dad and enjoyed playing in the sand and splashing in the water.

Today, Bouchra wanted to explore the ocean. Her parents said, "No, Bouchra. The ocean is too dangerous. You must stay close and be careful, okay?"

Bouchra said, "Okay!" and she began to walk away. But then she remembered something her mom had said, "Be careful!"

So she decided to stay close to her parents and not get too close to the water.

---

## Failure modes

What we found matches the central finding of the
[TinyStories paper](https://arxiv.org/abs/2305.07759): fluency, grammar, and local consistency emerge at small scale, while world knowledge and multi-step coherence do not.

**Semantic drift over long spans.** The story starts on-topic and then wanders
into an unrelated frame it can't get back from. This may be because stories are packed in
512-token windows without intra-document masking, so during training the model saw one story's tokens attending back into an unrelated story, this is a paper that noted this exacted problem
([Zhao et al., 2024](https://arxiv.org/abs/2402.13991)):

> **Amine has beaten his friend at school and his parents were called by the director.** She was determined to marry him as well, so she asked her parents if it was possible. […] The lawyer and Amine went together to church and were married.

**Pronoun and entity-state instability.** Gender agreement flips mid-story, for example a
male-named subject becomes "she" a sentence later.

**Physical implausibility.** The model has story grammar but no world model:

> This was his chance to become a helicopter too! […] He had become a helicopter!

**Inherited mojibake.** Some samples emit `â€œ` sequences. These come
from mis-encoded rows in the source dataset; `fix_mojibake` in `src/data.py`
catches most but not all, so a few corrupted byte sequences made it into the
BPE vocabulary. See the Known issues section of the README.
