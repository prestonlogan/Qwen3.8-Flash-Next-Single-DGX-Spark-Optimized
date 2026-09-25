# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified 2026 by Preston Logan for Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (see docs/OPTIMIZATIONS.md);
# original file distributed under Apache-2.0 by the vLLM project via MiaAI-Lab's overlay. Modifications under the same license.
"""Inference-only Qwen3.8-Flash-Next MTP (Multi-Token Predictor) model.

The MTP draft model reuses the Qwen3.8-Flash-Next backbone (PLE/HC/MoE) but:
  - drops all multi-modal handling (text-only),
  - forces PLE off while keeping the main model's HC stream count,
  - fuses the backbone hidden and the new-token embedding via
    ``residual_linear_shared`` (fc_embedding + shared fc_hidden) instead of
    the ``Linear(2H, H)`` + repeat used by other MTP variants,
  - emits TWO hidden streams per step (scheme A): a single stream [T, H]
    (final-mixer collapsed, fed to the LM head) and a pre-final-mixer
    multi stream [T, hc_count*H] (fed to the next draft step).
"""

from collections.abc import Iterable

import regex as re
import torch
from torch import nn

import os

from vllm.compilation.decorators import support_torch_compile
from vllm.logger import init_logger
from vllm.config import VllmConfig, replace, set_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    get_draft_quant_config,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_8_flash_next import (
    Qwen3_8FlashNextTextConfig,
)

from .hyperconnection import GatedResidual, HyperConnectionConfig
from .low_latency_gemm import enable_qwen38next_low_latency_gemm
from .model import (
    _HC_WEIGHTS_MAPPER,
    _QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES,
    Qwen3_8FlashNextDecoderLayer,
    Qwen3_8FlashNextMixtureOfExperts,
)


logger = init_logger(__name__)


# ---- EXPERIMENT (exp-qwen38fn-decode-speed): FP8 draft head -----------------
# Per-row E4M3 copy of the reduced draft head + Triton W8A16 GEMV (tactic from
# rwl4 / Mia PR #31, Apache-2.0, adapted from gabrielolympie sglang-flashnext-sm120).
# Draft-only: target logits, verification and rejection sampling are untouched.
import triton
import triton.language as tl


@triton.jit
def _exp_fp8_head_kernel(X, W, S, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         XM: tl.constexpr, BM: tl.constexpr):
    rn = tl.program_id(0) * 32 + tl.arange(0, 32)
    rm = tl.arange(0, BM)
    acc = tl.zeros((BM, 32), tl.float32)
    for k0 in tl.range(0, K, 256, num_stages=3):
        rk = k0 + tl.arange(0, 256)
        x = tl.load(X + rm[:, None] * XM + rk[None, :],
                    (rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        w = tl.load(W + rn[:, None] * K + rk[None, :],
                    (rn[:, None] < N) & (rk[None, :] < K), other=0.0).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    scale = tl.load(S + rn, rn < N, other=0.0)
    acc = acc * scale[None, :]
    tl.store(O + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))


@torch.no_grad()
def _exp_quantize_rows(weight: torch.Tensor):
    packed = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scale = torch.empty(weight.shape[0], dtype=torch.float32, device=weight.device)
    rows = max(1, (32 * 1024 * 1024) // (weight.shape[1] * 4))
    for start in range(0, weight.shape[0], rows):
        chunk = weight[start:start + rows].float()
        sc = (chunk.abs().amax(dim=1) / 448).clamp_min(1e-12)
        packed[start:start + rows] = (chunk / sc[:, None]).clamp(-448, 448).to(packed.dtype)
        scale[start:start + rows] = sc
    return packed, scale


def _exp_fp8_logits(x: torch.Tensor, fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.bfloat16).contiguous()
    m, k = x.shape
    n = fp8.shape[0]
    out = x.new_empty((m, n))
    _exp_fp8_head_kernel[(triton.cdiv(n, 32),)](x, fp8, scale, out, m, n, k, x.stride(0),
                                                max(16, triton.next_power_of_2(m)), num_warps=4)
    return out
# ---------------------------------------------------------------------------


def _attach_draft_vocab(model: nn.Module) -> None:
    """Slice the drafter's lm_head down to VLLM_MTP_DRAFT_VOCAB's token ids.

    The file is one integer token id per line (see files/build_draft_vocab.py).
    Leaves the full head in place and untouched; only get_top_tokens below
    reads the slice.
    """
    path = os.environ.get("VLLM_MTP_DRAFT_VOCAB", "").strip()
    if not path:
        return
    lm_head = getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.dim() != 2:
        logger.warning("MTP draft vocab: no 2-D lm_head weight; skipping.")
        return
    if getattr(lm_head, "tp_size", 1) != 1:
        logger.warning("MTP draft vocab: TP>1 is unsupported; skipping.")
        return
    scale = getattr(model.logits_processor, "scale", 1.0)
    if scale <= 0.0:
        logger.warning("MTP draft vocab: non-positive logit scale; skipping.")
        return

    org_vocab = int(getattr(lm_head, "org_vocab_size", weight.shape[0]))
    with open(path) as handle:
        ids = sorted({int(line) for line in handle if line.strip()})
    ids = [i for i in ids if 0 <= i < org_vocab]
    if not ids or len(ids) >= org_vocab:
        logger.warning(
            "MTP draft vocab: %d usable ids against a %d vocabulary; skipping.",
            len(ids), org_vocab,
        )
        return

    index = torch.tensor(ids, dtype=torch.long, device=weight.device)
    model.register_buffer(
        "_draft_lm_head_weight",
        weight.data.index_select(0, index).contiguous(),
        persistent=False,
    )
    model.register_buffer(
        "_draft_id_to_target_id",
        index.to(torch.int32),
        persistent=False,
    )
    if os.environ.get("EXP_MTP_DRAFT_HEAD_FP8", "0") == "1":
        packed, scales = _exp_quantize_rows(model._draft_lm_head_weight)
        model.register_buffer("_draft_lm_head_fp8", packed, persistent=False)
        model.register_buffer("_draft_lm_head_fp8_scale", scales, persistent=False)
        # Drop the BF16 slice: get_top_tokens only sees <= max_num_seqs rows.
        model._draft_lm_head_weight = torch.empty(0, device=weight.device, dtype=weight.dtype)
        for m in (1, 2, 3, 4, 8, 16):
            _exp_fp8_logits(torch.zeros((m, weight.shape[1]), dtype=torch.bfloat16,
                                        device=weight.device), packed, scales)
        torch.cuda.synchronize()
        logger.info("EXP FP8 draft head: %d rows, %.3f GiB", packed.shape[0],
                    (packed.numel() + scales.numel() * 4) / 2**30)
    if os.environ.get("EXP_DRAFT_MASK", "0") == "1":
        # In-place-swappable subset mask (0 = allowed, -inf = masked) so one launch can
        # measure acceptance for many draft vocabularies inside this superset.
        model.register_buffer(
            "_exp_draft_bias",
            torch.zeros(index.numel(), dtype=torch.bfloat16, device=weight.device),
            persistent=False,
        )
        logger.info("EXP draft mask buffer attached (%d rows)", index.numel())
    full_gib = weight.numel() * weight.element_size() / 2**30
    cut_gib = model._draft_lm_head_weight.numel() * weight.element_size() / 2**30
    logger.info(
        "MTP draft vocab: %d of %d tokens (%.1f%%); draft lm_head %.2f -> %.2f "
        "GiB per draft step, %.2f GiB saved per step at MTP %d",
        len(ids), org_vocab, 100.0 * len(ids) / org_vocab, full_gib, cut_gib,
        (full_gib - cut_gib) * 3, 3,
    )


def _remap_ignored_layers(
    ignored_layers: list[str],
    mtp_start_layer_idx: int,
) -> list[str]:
    remapped: list[str] = []
    for name in ignored_layers:
        if name.startswith("mtp."):
            new_name = re.sub(
                r"(?<=\.layers\.)\d+",
                lambda m: str(mtp_start_layer_idx + int(m.group(0))),
                name,
            )
            remapped.append(new_name)
        else:
            remapped.append(name)
    return remapped


def _remap_mtp_weight_name(name: str) -> str | None:
    """Map Qwen3.8-Flash-Next checkpoint paths into the standalone draft model."""

    for checkpoint_prefix in (
        "model.language_model.",
        "language_model.",
    ):
        if name.startswith(checkpoint_prefix):
            name = name.removeprefix(checkpoint_prefix)
            break

    if name.startswith("embed_tokens."):
        name = f"model.{name}"
    if name.startswith("model.mtp."):
        name = name.removeprefix("model.")
    if name.startswith("mtp.shared_head.head."):
        return name.replace("mtp.shared_head.head.", "lm_head.", 1)
    if name.startswith("model.shared_head.head."):
        return name.replace("model.shared_head.head.", "lm_head.", 1)
    if name.startswith("shared_head.head."):
        return name.replace("shared_head.head.", "lm_head.", 1)
    if name.startswith("model.lm_head."):
        return name.removeprefix("model.")
    if name.startswith("mtp."):
        return name.replace("mtp.", "model.", 1)
    if name.startswith("model.embed_tokens.") or name.startswith("lm_head."):
        return name
    return None


def _make_draft_vllm_config(
    vllm_config: VllmConfig,
    mtp_start_layer_idx: int,
) -> VllmConfig:
    """Ensure that the draft model config is set in the vLLM config."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_model_config is None:
        raise ValueError("speculative_config.draft_model_config must be set")

    draft_quant_config = get_draft_quant_config(vllm_config)

    # inject packed and ignored modules to the quantization config of draft model
    if draft_quant_config is not None:
        configure_quant_config(draft_quant_config, Qwen3_8FlashNextMTP)
        ignored_layers = getattr(draft_quant_config, "ignored_layers", None)
        if ignored_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "ignored_layers",
                _remap_ignored_layers(ignored_layers, mtp_start_layer_idx),
            )
        exclude_modules = getattr(draft_quant_config, "exclude_modules", None)
        if exclude_modules:
            setattr(  # noqa: B010
                draft_quant_config,
                "exclude_modules",
                _remap_ignored_layers(exclude_modules, mtp_start_layer_idx),
            )

    draft_vllm_config = replace(
        vllm_config,
        model_config=speculative_config.draft_model_config,
    )
    # VllmConfig post-init derives the target quant config, so restore the
    # independently resolved draft quant config after replacement.
    draft_vllm_config.quant_config = draft_quant_config
    return draft_vllm_config


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_8FlashNextMultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _HC_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        model_config = vllm_config.model_config
        config: Qwen3_8FlashNextTextConfig = model_config.hf_text_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        draft_vllm_config = _make_draft_vllm_config(
            vllm_config,
            self.mtp_start_layer_idx,
        )
        with set_current_vllm_config(draft_vllm_config, prefix=prefix):
            # residual_linear_shared fusion: fc_embedding projects the token
            # embedding, fc_hidden (shared across HC branches) projects the
            # backbone hidden; the embedding is added as a residual to every
            # branch (see mtp_residual_linear_shared.md).
            self.fc_embedding = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_embedding",
            )
            self.fc_hidden = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_hidden",
            )
            self.layers = nn.ModuleList(
                Qwen3_8FlashNextDecoderLayer(
                    draft_vllm_config,
                    layer_type="full_attention",
                    prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}",
                )
                for idx in range(self.num_mtp_layers)
            )

        self.pre_fc_norm_embedding = GemmaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            self.hidden_size * self.hc_count, eps=config.rms_norm_eps
        )
        # HC final mixer collapses the multi stream into [T, H] for the LM head.
        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = GatedResidual(
            hc_config,
            use_combine=False,
            prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size * self.hc_count
        )

    def _iter_qsa_attentions(self):
        """Yield MTP attention modules that own a QSA indexer."""

        for layer in self.layers:
            attention = getattr(layer, "self_attn", None)
            if (
                attention is not None
                and getattr(attention, "indexer", None) is not None
            ):
                yield attention

    def set_skip_topk(self, skip: bool) -> None:
        """Select on MTP step 0 and reuse its QSA indices on later steps."""

        if not getattr(self, "_mtp_index_share_logged", False):
            self._mtp_index_share_logged = True
            logger.info(
                "MTP index share: index_share_for_mtp_iteration ACTIVE "
                "(set_skip_topk reached the draft QSA indexer)"
            )
        for attention in self._iter_qsa_attentions():
            attention.indexer.skip_topk = skip

    def compact_topk_indices(self, row_indices: torch.Tensor) -> None:
        """Keep each request's target-aligned step-0 sparse-index row."""

        num_rows = row_indices.numel()
        for attention in self._iter_qsa_attentions():
            buffer = attention.topk_indices_buffer
            selected = buffer.index_select(0, row_indices)
            buffer[:num_rows].copy_(selected)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        hc_count = self.hc_count
        hidden_size = self.hidden_size

        if get_pp_group().is_first_rank:
            assert hidden_states is not None
            if inputs_embeds is None:
                assert input_ids is not None
                inputs_embeds = self.embed_input_ids(input_ids)
            # Embedding branch: pre-norm -> fc_embedding -> [T, H].
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            inputs_embeds = self.fc_embedding(inputs_embeds)

            # Backbone hidden is multi-stream [T, hc_count*H] (scheme A:
            # the main model truly emits the pre-final-mixer multi stream
            # on the first step; subsequent steps reuse the prior draft
            # step's multi stream).
            num_tokens = hidden_states.shape[0]
            hidden_states = hidden_states.view(num_tokens, hc_count, hidden_size)
            hidden_states = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(
                num_tokens, hc_count, hidden_size
            )
            hidden_states = self.fc_hidden(hidden_states)
            # Add the embedding residual to every branch, then fold back
            # to [T, hc_count*H] (HC outer, HS inner) for the HC decoder.
            hidden_states = inputs_embeds.unsqueeze(-2) + hidden_states
            hidden_states = hidden_states.flatten(-2)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer = self.layers[current_step_idx]
        hidden_states, block_output, injection = layer(
            hidden_states=hidden_states,
            prev_block_output=None,
            prev_injection=None,
            positions=positions,
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
        )
        if not get_pp_group().is_last_rank:
            # As in the target model, PP carries a materialized tensor rather
            # than the delayed hidden/output/injection tuple.
            hidden_states = layer.mlp_hyper_connection.combine(
                hidden_states, block_output, injection
            )
            return IntermediateTensors({"hidden_states": hidden_states})

        # Last PP rank finalize. Keep both:
        #   (A) sample_hidden_states [T, H]  -> single stream for the LM head
        #   (B) multi_hidden [T, hc_count*H] -> pre-final-mixer multi stream
        #       for the next draft step (zero extra compute, just kept).
        multi_hidden, sample_hidden_states, _ = (
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states, block_output, injection
            )
        )
        return sample_hidden_states, multi_hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(
            self,
            skip_substrs=["hyper_connection_mixer.block_inject_weight"],
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_8FlashNextMTP(nn.Module, SupportsPP, Qwen3_8FlashNextMixtureOfExperts):
    @staticmethod
    def checkpoint_weight_filter(name: str) -> bool:
        return _remap_mtp_weight_name(name) is not None

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3_8FlashNextMTP currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen3_8FlashNextMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)
        enable_qwen38next_low_latency_gemm(self, vllm_config.model_config.dtype)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy draft token ids, over the reduced vocabulary when present.

        Positive logit scaling and the tanh soft cap are both monotonic, so the
        argmax is unchanged by skipping them; _attach_draft_vocab refuses to
        engage when the scale is not positive.
        """
        weight = getattr(self, "_draft_lm_head_weight", None)
        if weight is None:
            return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
        fp8 = getattr(self, "_draft_lm_head_fp8", None)
        if fp8 is not None:
            if hidden_states.shape[0] > 32:
                full = self.logits_processor(self.lm_head, hidden_states)
                logits = full.index_select(-1, self._draft_id_to_target_id.long())
            else:
                logits = _exp_fp8_logits(hidden_states, fp8, self._draft_lm_head_fp8_scale)
            bias = getattr(self, "_exp_draft_bias", None)
            if bias is not None:
                logits = logits + bias
            return self._draft_id_to_target_id[logits.argmax(dim=-1)].to(torch.long)
        logits = torch.nn.functional.linear(hidden_states.to(weight.dtype), weight)
        bias = getattr(self, "_exp_draft_bias", None)
        if bias is not None:
            logits = logits + bias
        return self._draft_id_to_target_id[logits.argmax(dim=-1)].to(torch.long)


    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # T8: optional experimental MTP weight patch (LoRA-merged drafter tensors, checkpoint names).
        _t8_patch = {}
        _t8_path = os.environ.get("EXP_MTP_PATCH", "").strip()
        if _t8_path:
            from safetensors.torch import load_file as _t8_load
            _t8_patch = _t8_load(_t8_path)
            logger.warning("T8: MTP weight patch %s with %d tensors", _t8_path, len(_t8_patch))
        _t8_used = set()

        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    for ck in (name, name.removeprefix("model.")):
                        if ck in _t8_patch:
                            weight = _t8_patch[ck].to(weight.dtype); _t8_used.add(ck); break
                    yield remapped_name, weight

        loader = AutoWeightsLoader(
            self,
            skip_substrs=["hyper_connection_mixer.block_inject_weight"],
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
        )
        loaded = loader.load_weights(remap_weight_names())
        if _t8_patch:
            logger.warning("T8: applied %d/%d patch tensors; unused=%s", len(_t8_used), len(_t8_patch), sorted(set(_t8_patch) - _t8_used)[:5])
        _attach_draft_vocab(self)
        return loaded


__all__ = ["Qwen3_8FlashNextMTP", "Qwen3_8FlashNextMultiTokenPredictor"]
