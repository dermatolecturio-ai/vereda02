# Initial Prototype Audit

Date: 2026-06-04

## Positive Findings

- A working byte-level training and C++ inference prototype exists.
- The project already owns a large Brazilian Portuguese corpus.
- Ternary export and stateful inference were explored concretely.
- The research documents converge on patching, recurrence, persistent state,
  explicit evaluation, and episodic memory.

## Negative Findings

- The base model generated only spaces for a simple conversational prompt.
- Training credit assignment was truncated to 8 bytes.
- Only about 16.35 MB of the roughly 15 GB local corpus was loaded.
- The 200M runtime took about 2.05 seconds to generate one byte on Apple
  Silicon with the current dense ternary loops.
- Lorentz projection existed in C++ inference but not in training.
- There was no held-out evaluation suite or reproducible baseline comparison.

## Lessons

- Change one architectural variable at a time.
- Establish a functional baseline before adding geometry or quantization.
- Treat negative results as evidence and preserve them with configs and seeds.
- Conversation quality requires conversational data and explicit evaluation.

