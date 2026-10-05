# Security

Report vulnerabilities privately to savagetism@icloud.com — do not open
public issues for exploitable weaknesses.

Scope: protobuf wire encoding correctness, weight blob layout, manifest
emission, IR validation bypass (a check the compiler promises but does
not enforce).

Out of scope: graphs hand-built through `Block::op` (see THREAT_MODEL.md
— raw binding intentionally bypasses IR checks), semantic correctness of
caller-described models, and CoreML/`coremlc` platform behavior itself.
