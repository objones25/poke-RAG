"""Compatibility shims for FlagEmbedding 1.3.5 with transformers 5.x.

Background
==========
FlagEmbedding 1.3.5 was built against transformers 4.x. When transformers 5.x
landed, two critical internal APIs were removed or significantly refactored.
FlagEmbedding still calls these APIs at import time and during tokenization,
causing immediate import failure without these shims.

Why This File Exists
====================
Without _compat.py patching, importing BGEEmbedder or BGEReranker crashes with:
  - AttributeError: module 'transformers.utils.import_utils' has no attribute
    'is_torch_fx_available'
  - AttributeError: 'PreTrainedTokenizerBase' object has no attribute
    'prepare_for_model'

Patched APIs
============
1. **transformers.utils.import_utils.is_torch_fx_available** (removed in 5.x)
   - Was: Function to check whether torch.fx is importable
   - Removed: Code using FX tracing was refactored out entirely
   - FlagEmbedding calls it during model initialization
   - Shim: Probe torch.fx import; return True/False

2. **PreTrainedTokenizerBase.prepare_for_model** (removed in 5.x)
   - Was: Method to combine pre-tokenized IDs with special tokens (BOS/EOS)
     and optionally truncate to max_length
   - Removed: Folded into __call__() and no longer exposed as a standalone method
   - FlagEmbedding calls it: The reranker tokenization pipeline calls
     tokenizer.prepare_for_model(ids, pair_ids, ...) to prepare cross-encoder inputs
   - Implementation note: build_inputs_with_special_tokens was also removed in
     transformers 5.x. Rather than creating a third patch, its logic is inlined
     directly inside the prepare_for_model shim (see docstring below)
   - Shim: Re-implement the full method with XLMRobertaTokenizer special-token
     pattern reconstruction

Guard Pattern
=============
Both patches check `if not hasattr(...)` before patching. This means:
  - Re-importing this module is safe and idempotent
  - If FlagEmbedding is updated to restore the APIs, the shim won't override them
  - The shim only patches what is missing

Import Order Requirement
=========================
This module MUST be imported before FlagEmbedding is first imported.
  - embedder.py and reranker.py each start with: import src.retrieval._compat
  - Do not remove these imports or reorder module initialization
"""

from transformers.utils import import_utils as _tf_import_utils

# Patch 1: transformers.utils.import_utils.is_torch_fx_available
# ==============================================================
# Transformers 5.x removed this function after removing torch.fx tracing support.
# FlagEmbedding calls it during model initialization (inside BGEM3FlagModel).
# The patch is idempotent: only applies if the function doesn't already exist.
if not hasattr(_tf_import_utils, "is_torch_fx_available"):

    def _is_torch_fx_available() -> bool:
        """Check whether torch.fx is importable.

        Returns True if torch.fx can be imported, False otherwise.
        This mimics the original function signature and behavior.
        """
        try:
            import torch.fx  # noqa: F401

            return True
        except ImportError:
            return False

    _tf_import_utils.is_torch_fx_available = _is_torch_fx_available  # type: ignore[attr-defined]


from transformers import BatchEncoding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Patch 2: PreTrainedTokenizerBase.prepare_for_model
# ==================================================
# Transformers 5.x removed this method from the public tokenizer API.
# It was folded into __call__() and is no longer directly accessible.
# FlagEmbedding's reranker pipeline calls tokenizer.prepare_for_model(ids, pair_ids, ...)
# to prepare cross-encoder inputs. This patch restores the full implementation.
#
# Implementation note: build_inputs_with_special_tokens was also removed in
# transformers 5.x. Its logic is inlined directly in the special-token section
# below (see lines with XLMRobertaTokenizer pattern) rather than creating a
# separate shim function.
#
# The patch is idempotent: only applies if the method doesn't already exist.
if not hasattr(PreTrainedTokenizerBase, "prepare_for_model"):

    def _prepare_for_model(
        self: PreTrainedTokenizerBase,
        ids: list[int],
        pair_ids: list[int] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool | str | None = None,
        max_length: int | None = None,
        **_kwargs: object,
    ) -> BatchEncoding:
        """Combine pre-tokenized IDs with special tokens and optionally truncate.

        This method is called by FlagEmbedding's reranker to prepare inputs for
        the cross-encoder model (BGE-M3 reranker uses XLMRobertaTokenizer).

        Args:
            ids: Pre-tokenized integer IDs (not raw text). E.g., [101, 2054, 2003, 102]
            pair_ids: Optional second sequence IDs for pair encoding (cross-encoder use).
                      E.g., [101, 1045, 2572, 102] for "I love"
            add_special_tokens: If True, combine IDs with BOS/EOS tokens following
                                the tokenizer's special-token pattern (see below).
                                If False, return ids + pair_ids as-is.
            padding: Ignored (legacy parameter). FlagEmbedding doesn't use padding here.
            truncation: Truncation strategy if combined length > max_length:
                        - "only_second": Truncate only pair_ids (keep ids complete)
                        - "longest_first" or True: Truncate from the longer sequence
            max_length: Maximum total length for input_ids (including special tokens).
                        If None, no truncation is applied.
            **_kwargs: Ignored (legacy parameters for compatibility)

        Returns:
            BatchEncoding with 'input_ids' and 'attention_mask':
              - 'input_ids': Combined token IDs (with special tokens if add_special_tokens)
              - 'attention_mask': All 1s (no padding, all tokens are real)

        Special Token Pattern (XLMRobertaTokenizer via BGE-M3 reranker):
        ===============================================================
        XLMRobertaTokenizer follows the RoBERTa pair encoding convention:
          - Single sequence:
            [BOS(id=0)] + ids + [EOS(id=2)]
            Example: [0, 101, 2054, 102] for input IDs [101, 2054, 102]

          - Pair sequence (ids + pair_ids):
            [BOS(id=0)] + ids + [EOS(id=2), EOS(id=2)] + pair_ids + [EOS(id=2)]
            Example: [0, 101, 2054, 102, 2, 101, 1045, 2572, 102, 2]
                     for ids [101, 2054, 102] and pair_ids [101, 1045, 2572, 102]

        Note: BOS/EOS token IDs are read from self.bos_token_id and self.eos_token_id,
              guarding against None values. This is why the shim inlines
              build_inputs_with_special_tokens logic: we need to handle both cases.

        Note on FlagEmbedding usage:
        ===========================
        FlagEmbedding only reads ['input_ids'] from the returned BatchEncoding.
        It does not use 'attention_mask' (no padding needed for pre-tokenized inputs).
        The attention_mask is set to all-1s for completeness.
        """
        # Convert to mutable lists for truncation (if needed)
        ids = list(ids)
        pair_ids = list(pair_ids) if pair_ids is not None else None

        # Truncation logic: Only applies if max_length is set
        if max_length is not None:
            # Calculate how many special tokens will be added (e.g., BOS, EOS)
            # This is needed to know how much room is left for content IDs.
            n_special = (
                self.num_special_tokens_to_add(pair=pair_ids is not None)
                if add_special_tokens
                else 0
            )

            # Strategy 1: Truncate only pair_ids (keep first sequence complete)
            # Used in cross-encoder scenarios where the query is fixed and the
            # candidate document is truncated to fit the budget.
            if truncation == "only_second" and pair_ids is not None:
                allowed = max(0, max_length - len(ids) - n_special)
                pair_ids = pair_ids[:allowed]

            # Strategy 2: Truncate from longest first (default if truncation=True)
            # Greedy approach: trim tokens from whichever sequence is longer until
            # total length (including special tokens) fits within max_length.
            elif truncation in ("longest_first", True):
                while len(ids) + (len(pair_ids) if pair_ids else 0) + n_special > max_length:
                    if pair_ids and len(pair_ids) >= len(ids):
                        # Pair is longer or equal, trim it from the end
                        pair_ids = pair_ids[:-1]
                    elif ids:
                        # First sequence is longer, trim it from the end
                        ids = ids[:-1]
                    else:
                        # Both exhausted, break to avoid infinite loop
                        break

        # Special-token insertion: Inline implementation of build_inputs_with_special_tokens
        # This logic was removed in transformers 5.x, so we implement it here directly.
        # XLMRobertaTokenizer (used by BGE-M3 reranker) uses RoBERTa pair convention.
        if add_special_tokens:
            # Extract BOS and EOS token IDs from the tokenizer instance
            # These are typically: bos_token_id=0 (BOS), eos_token_id=2 (EOS)
            # Guard against None in case tokenizer doesn't define them
            bos = [self.bos_token_id] if self.bos_token_id is not None else []
            eos = [self.eos_token_id] if self.eos_token_id is not None else []

            # Apply RoBERTa pair encoding pattern:
            # - Single: [BOS] + ids + [EOS]
            # - Pair:   [BOS] + ids + [EOS, EOS] + pair_ids + [EOS]
            #
            # The double [EOS, EOS] in pair encoding marks the boundary between
            # the first and second sequences. This is the RoBERTa convention.
            if pair_ids is not None:
                input_ids = bos + ids + eos + eos + pair_ids + eos
            else:
                input_ids = bos + ids + eos
        else:
            # No special tokens: concatenate sequences as-is
            input_ids = ids + (pair_ids or [])

        # Return BatchEncoding with input_ids and attention_mask
        # Note: attention_mask is all-1s (no padding needed for pre-tokenized inputs)
        # FlagEmbedding only reads ['input_ids'] from this response; attention_mask
        # is included for completeness and API compatibility.
        return BatchEncoding({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})

    # Attach the shim method to the tokenizer base class (idempotent via hasattr check above)
    PreTrainedTokenizerBase.prepare_for_model = _prepare_for_model  # type: ignore[attr-defined]
