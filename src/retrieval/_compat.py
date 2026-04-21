# Compatibility shims for FlagEmbedding 1.3.5 + transformers 5.x.
#
# FlagEmbedding 1.3.5 uses two APIs removed in transformers 5.x:
#   1. is_torch_fx_available  (removed from transformers.utils.import_utils)
#   2. PreTrainedTokenizerBase.prepare_for_model  (removed; folded into __call__)
#
# Both are restored here before FlagEmbedding is first imported.

from transformers.utils import import_utils as _tf_import_utils

if not hasattr(_tf_import_utils, "is_torch_fx_available"):

    def _is_torch_fx_available() -> bool:
        try:
            import torch.fx  # noqa: F401

            return True
        except ImportError:
            return False

    _tf_import_utils.is_torch_fx_available = _is_torch_fx_available  # type: ignore[attr-defined]


from transformers import BatchEncoding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

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
        """Restored for FlagEmbedding 1.3.5 compatibility with transformers 5.x.

        Takes pre-tokenised IDs (not text), combines them with special tokens,
        and optionally truncates.  FlagEmbedding only reads ['input_ids'].
        """
        ids = list(ids)
        pair_ids = list(pair_ids) if pair_ids is not None else None

        if max_length is not None:
            n_special = (
                self.num_special_tokens_to_add(pair=pair_ids is not None)
                if add_special_tokens
                else 0
            )
            if truncation == "only_second" and pair_ids is not None:
                allowed = max(0, max_length - len(ids) - n_special)
                pair_ids = pair_ids[:allowed]
            elif truncation in ("longest_first", True):
                while len(ids) + (len(pair_ids) if pair_ids else 0) + n_special > max_length:
                    if pair_ids and len(pair_ids) >= len(ids):
                        pair_ids = pair_ids[:-1]
                    elif ids:
                        ids = ids[:-1]
                    else:
                        break

        if add_special_tokens:
            # build_inputs_with_special_tokens removed in transformers 5.x.
            # XLMRobertaTokenizer (used by BGE-M3 reranker) follows RoBERTa pattern:
            # single: [BOS] + ids + [EOS]
            # pair:   [BOS] + ids1 + [EOS, EOS] + ids2 + [EOS]
            bos = [self.bos_token_id] if self.bos_token_id is not None else []
            eos = [self.eos_token_id] if self.eos_token_id is not None else []
            if pair_ids is not None:
                input_ids = bos + ids + eos + eos + pair_ids + eos
            else:
                input_ids = bos + ids + eos
        else:
            input_ids = ids + (pair_ids or [])

        return BatchEncoding({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})

    PreTrainedTokenizerBase.prepare_for_model = _prepare_for_model  # type: ignore[attr-defined]
