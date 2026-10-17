"""Recover complete objects with a unique validated dispatch; never repair kernel code."""
import hashlib
import json


def recover(content,schema,allowed_tools=()):
    stripped=content.lstrip()
    if not stripped.startswith('{'):
        return None
    try:
        value,end=json.JSONDecoder().raw_decode(stripped)
    except ValueError:
        return None
    if not isinstance(value,dict):
        return None
    candidates=[]
    try:
        candidates.append((schema.model_validate(value),None))
    except ValueError:
        # Only the operation tag may be inferred, and only when exactly one CURRENT
        # tool satisfies every field and conditional rule. Nothing in code_lines changes.
        if 'tool' not in value:
            for tool in allowed_tools:
                try:
                    candidates.append((schema.model_validate({**value,'tool':tool}),tool))
                except ValueError:
                    continue
    if len(candidates)!=1:
        return None
    model,inferred=candidates[0]
    prefix=stripped[:end]
    tail=stripped[end:]
    return model,{'mode':'unique_contract' if inferred else 'complete_object',
        'inferred_tool':inferred,'prefix_chars':end,'tail_chars':len(tail),
        'prefix_sha256':hashlib.sha256(prefix.encode()).hexdigest(),
        'tail_sha256':hashlib.sha256(tail.encode()).hexdigest(),
        'scope':'First fully closed JSON object only; unique allowed tool match. No field/source text is repaired; ambiguity/truncation rejected.'}
