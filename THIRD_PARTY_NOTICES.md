# Third-party sources

The AutoVerus and VeruSAGE implementations derive from Microsoft's [verus-proof-synthesis](https://github.com/microsoft/verus-proof-synthesis). Their MIT licenses and source copyright notices are retained under each `pipelines/` implementation. The bundled Lynette Rust sources retain their upstream notices and Cargo dependency declarations.

The initial-program collection uses the StarVerus benchmark, covering VerusBench, MBPP-verified, HumanEval-Verus, DAFNY2VERUS, and VeriCoding tasks. Embedded task comments and notices are preserved. The benchmark problem, a generated initial program, and a recorded repair state are different objects; the provided data describes the latter two and their benchmark task identifiers.

The prompt selector adapts the StarVerus benchmark planner's system prompt and routing semantics, with retrieved repair examples and outcome evidence. Its final prompt texts and the upstream license are in `scripts/prompts/`. The reference is Jiang et al., *StarVerus: LLM-Powered Multi-Agent Collaboration for Industrial Rust Code Verification Automation*, KDD 2026, doi:10.1145/3770855.3818485.

Frozen text vectors were produced with [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3). The encoder weights are obtained separately from their provider. The project-specific model draws on SASRec and Bayesian Personalized Ranking; see the accompanying paper for references. This snapshot retains existing upstream notices; it does not assign a new blanket license to third-party benchmark material.
