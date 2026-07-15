# ATLAS Hardcoded Belief Audit

## Principle

ATLAS may be given factual target definitions, chronology rules, sample controls and statistical safeguards. It must not be given manual beliefs about which baseball evidence matters or how strongly that evidence should influence a prediction.

## Audit results

- Modules inspected: 14
- Review finding rows: 104
- Review numeric constants: 66
- Modules with no prescriptive logic detected: 0
- Modules requiring focused review: 4
- Modules requiring detailed review: 10

## Interpretation

- `ALLOWED_FACTUAL`: factual historical labels and outcome definitions.
- `ALLOWED_SAFETY_STATISTICAL`: chronology, validation, sample and statistical-reliability controls.
- `REVIEW_PRESCRIPTIVE`: possible manually assigned influence or belief.
- `REVIEW_CONTEXT`: requires reading the surrounding function before use.

No flagged module should be connected to Phase 2E until its review rows have been classified as factual, statistical, learned from training data, or removed.