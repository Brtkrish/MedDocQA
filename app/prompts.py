"""
Three distinct prompting strategies for the same underlying task
(answer a question using retrieved context). Keeping them explicit and
separate - rather than one "smart" prompt - is what lets you actually
compare and evaluate them, which is the point of this project.
"""
from enum import Enum
from typing import List


class Strategy(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    STRUCTURED_COT = "structured_cot"


def _format_context(chunks: List[dict]) -> str:
    return "\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}"
        for c in chunks
    )


def build_prompt(strategy: Strategy, question: str, chunks: List[dict]) -> str:
    context = _format_context(chunks)

    if strategy == Strategy.ZERO_SHOT:
        return (
            "Answer the question using ONLY the context below. "
            "If the answer isn't in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    if strategy == Strategy.FEW_SHOT:
        examples = (
            "Example 1\n"
            "Context: [Source: trial_protocol.pdf, page 2] Patients must be "
            "aged 18-65 and have no history of renal impairment.\n"
            "Question: What is the age range for eligible patients?\n"
            "Answer: 18 to 65 years old (trial_protocol.pdf, page 2).\n\n"
            "Example 2\n"
            "Context: [Source: drug_label.pdf, page 1] Recommended dose is "
            "10mg once daily, taken with food.\n"
            "Question: What is the recommended dosage?\n"
            "Answer: 10mg once daily, taken with food (drug_label.pdf, page 1).\n\n"
        )
        return (
            "Answer the question using ONLY the provided context, following "
            "the style of the examples below - always cite source and page.\n\n"
            f"{examples}"
            f"Now answer using this context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    if strategy == Strategy.STRUCTURED_COT:
        return (
            "You will answer a question using only the context provided. "
            "Think step by step, then output your final answer as JSON.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Step 1: Identify which context passages (if any) are relevant.\n"
            "Step 2: Reason briefly about what they say.\n"
            "Step 3: Output your final answer.\n\n"
            "Respond in this exact JSON format and nothing else:\n"
            "{\n"
            '  "reasoning": "<1-2 sentence reasoning>",\n'
            '  "answer": "<final answer, or \'not found in context\'>",\n'
            '  "sources": ["<source, page>", ...]\n'
            "}"
        )

    raise ValueError(f"Unknown strategy: {strategy}")
