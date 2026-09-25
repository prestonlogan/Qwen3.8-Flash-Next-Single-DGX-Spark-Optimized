# Licensing and third-party material

| Path | Origin | License |
|---|---|---|
| everything not listed below (scripts, `overlays/runtime/exec_*.py`, `exp_*.py`, `bench/`, `docs/`, `t8_merge.py`, `exp_dmb.c`) | this project, © 2026 Preston Logan | AGPL-3.0-or-later (`LICENSE`) |
| `files/vllm/**` (`*_patched.py`, `ple_offload/*`) | vLLM source as patched by MiaAI Lab's recipe (commit `d038090`), byte-identical | Apache-2.0 (`LICENSES/Apache-2.0-vLLM.txt`); headers preserved |
| `overlays/runtime/mtp_exp_t8.py`, `ple_layer_fixrows.py`, `ple_worker_zc.py`, `short_conv_attn_exp.py` | the corresponding vLLM/Mia-overlay files, further modified by this project (header notes the modification) | Apache-2.0; modifications under the same license |
| `files/mia/build_ple_packed_table.py`, `scripts/memwatch.sh` | MiaAI Lab, © 2026 MiaAI Lab | AGPL-3.0-or-later |
| `files/chat-template/chat_template.jinja` | chat template as shipped in MiaAI Lab's recipe; its version tag names its upstream author ("froggeric") | as distributed with MiaAI Lab's recipe |
| `overlays/runtime/dv_cur_u48k.txt` | draft-vocabulary id list derived by this project from Mia's draft vocabulary and token statistics | AGPL-3.0-or-later |
| `adapters/r32a/lora_r32a.pt` | LoRA delta for the checkpoint's MTP drafter, trained by this project on self-generated text | AGPL-3.0-or-later for our contribution. It is only meaningful together with the checkpoint, whose weights are governed by their own license (Mia-AiLab model card and upstream Qwen terms). |

**Not included:**

- Model weights. The checkpoint is downloaded explicitly by `download.sh`.
- The merged 175 MB drafter patch. It is generated locally by `scripts/build_adapter.sh`.
- The packed PLE table (~27 GiB). It is generated locally by `scripts/prepare.sh`.

The Docker image `vllm/vllm-openai` (pinned by digest) and its contents are used as-is under their own licenses.

- `overlays/block_drop/patch_block_drop.py` — copied unmodified from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark `files/patch_block_drop.py` (PR #71, commit ffc41629c0a9aa9a390f18fecd6f1f19e7c3b024, author usmaneth), AGPL-3.0. It backports vllm-project/vllm#53388 (Apache-2.0).
