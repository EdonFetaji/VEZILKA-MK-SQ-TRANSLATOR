"""CPU smoke test of the trainer (plan Verification §6): a tiny random Gemma3 built locally (no network),
the 64-row ``--smoke`` dataset, 5 steps. Checks freezing, optimizer settings, keep/resume and the bf16
``model/`` reload."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("trl")

from src.pipeline import config as config_mod
from src.pipeline import trainer as tr
from src.pipeline.train import SMOKE_WORDS, run, smoke_dataset, with_max_steps

CHAT_TEMPLATE = (
    "{% for m in messages %}<start_of_turn>{{ m['role'] }} {{ m['content'] }}<end_of_turn>{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model{% endif %}"
)
SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>", "<img>", "<boi>", "<eoi>"]


def make_tiny_model(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration, PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate(SPECIALS + ["user", "assistant", "model", *SMOKE_WORDS])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>", unk_token="<unk>",
        additional_special_tokens=SPECIALS[4:],
    )
    hf_tok.chat_template = CHAT_TEMPLATE
    cfg = Gemma3Config(
        text_config={
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16, "max_position_embeddings": 512,
            "sliding_window": 64, "pad_token_id": 0, "eos_token_id": 1, "bos_token_id": 2,
        },
        vision_config={
            "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
            "image_size": 28, "patch_size": 14,
        },
        mm_tokens_per_image=4, image_token_index=6, boi_token_index=7, eoi_token_index=8,
    )
    torch.manual_seed(0)
    Gemma3ForConditionalGeneration(cfg).save_pretrained(path)
    hf_tok.save_pretrained(path)


def make_cfg(model_dir: Path, run_dir: Path, **training) -> config_mod.RunConfig:
    raw = {
        "run_id": "SMOKE",
        "seed": 1,
        "model": {
            "init_from": str(model_dir),
            "attn_implementation": "sdpa",
            "freeze": ["*embed_tokens*", "*vision_tower*", "*multi_modal_projector*"],
        },
        "data": {"max_length": 64, "packing": True},
        "training": {
            "max_steps": 5, "micro_batch_rows": 4, "micro_batch_tokens": 256, "grad_accum": 2,
            "tokens_per_step": 512, "dev_loss_every_steps": 2,
            "lr_schedule": {"type": "wsd", "peak": 1e-3, "min": 1e-4, "warmup_start": 1e-4,
                            "warmup_steps": 1, "constant_end_step": 3, "cooldown_steps": 2},
            "save": {"every_steps": 2, "keep_steps": [3], "with_optimizer": True},
            "logging": {"every_steps": 1},
            **training,
        },
        "output": {"local_dir": str(run_dir)},
    }
    return config_mod.parse(raw)  # parse() merges the optimizer/lr_schedule/save/resume defaults


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("tiny-gemma3")
    make_tiny_model(path)
    return path


def test_freeze_and_optimizer(model_dir, tmp_path):
    cfg = make_cfg(model_dir, tmp_path / "runs" / "SMOKE")
    model, _proc = tr.load_model_and_tokenizer(cfg, tmp_path / "runs" / "SMOKE")
    assert next(model.parameters()).dtype == torch.float32
    names = {n: p.requires_grad for n, p in model.named_parameters()}
    assert not any(v for n, v in names.items() if "embed_tokens" in n or "vision_tower" in n)
    assert not any(v for n, v in names.items() if "multi_modal_projector" in n)
    assert any(v for n, v in names.items() if "language_model.layers" in n)
    opt = tr.build_optimizer(cfg, model)
    d = opt.defaults
    assert d["beta1"] is None and d["decay_rate"] == -0.8 and d["clip_threshold"] == 1.0
    assert d["eps"] == (1e-30, 1e-3) and d["lr"] == 1e-3
    assert not (d["scale_parameter"] or d["relative_step"] or d["warmup_init"]) and d["weight_decay"] == 0.0
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert {id(p) for g in opt.param_groups for p in g["params"]} == trainable


def test_train_keep_resume_and_reload(model_dir, tmp_path):
    run_dir = tmp_path / "runs" / "SMOKE"
    cfg = make_cfg(model_dir, run_dir)

    # 1) a 3-step run, as if it crashed after step 3 (checkpoints at 2 and 3; 3 is also a keep step)
    assert run(with_max_steps(cfg, 3), run_dir, smoke=True) == 0
    ckpts = run_dir / "checkpoints"
    assert (ckpts / "checkpoint-3").is_dir()
    keep = ckpts / "keep" / "step-000003"
    assert (keep / "optimizer.pt").is_file() and (keep / "model.safetensors").is_file()

    # 2) the full 5-step run resumes from checkpoint-3 instead of starting over
    assert run(cfg, run_dir, smoke=True) == 0
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "done" and status["step"] == 5 and status["tokens"] > 0
    state = json.loads((ckpts / "checkpoint-4" / "trainer_state.json").read_text())
    assert state["global_step"] == 4
    assert sorted(p.name for p in ckpts.glob("checkpoint-*")) == ["checkpoint-4", "checkpoint-5"]
    train_log = [json.loads(x) for x in (run_dir / "metrics" / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in train_log] == [1, 2, 3, 4, 5]  # the rerun resumed at 3, it did not restart
    assert (run_dir / "metrics" / "dev_loss.jsonl").read_text().strip()
    lr_by_step = {r["step"]: r["learning_rate"] for r in train_log}
    assert lr_by_step[2] == pytest.approx(1e-3)  # lr logged at step k is lr_at(k - 1)

    # 3) the bf16 model/ reloads
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    m = AutoModelForImageTextToText.from_pretrained(run_dir / "model")
    assert m.dtype == torch.bfloat16
    tok = AutoTokenizer.from_pretrained(run_dir / "model")
    ids = tok(" ".join(SMOKE_WORDS[:4]), return_tensors="pt")
    assert m(**ids).logits.shape[-1] == 64


def test_branch_from_keep(model_dir, tmp_path):
    """A branch loads the parent's keep/step-N weights + optimizer and starts the schedule there."""
    runs = tmp_path / "runs"
    parent = make_cfg(model_dir, runs / "PARENT")
    assert run(with_max_steps(parent, 3), runs / "PARENT", smoke=True) == 0

    sched = {"type": "cooldown", "peak": 1e-3, "min": 1e-4, "constant_end_step": 3, "cooldown_steps": 2}
    resume = {"from_run": "PARENT", "from_step": 3, "optimizer_state": True, "start_global_step": 3}
    branch = make_cfg(model_dir, runs / "BRANCH", max_steps=2, lr_schedule=sched, resume=resume,
                      save={"every_steps": None, "keep_steps": [], "with_optimizer": False})
    model, proc = tr.load_model_and_tokenizer(branch, runs / "BRANCH")
    from transformers import AutoModelForImageTextToText

    kept = dict(AutoModelForImageTextToText.from_pretrained(
        runs / "PARENT" / "checkpoints" / "keep" / "step-000003").named_parameters())
    fresh = dict(AutoModelForImageTextToText.from_pretrained(model_dir).named_parameters())
    name, p = next((n, p) for n, p in model.named_parameters() if p.requires_grad)
    assert torch.equal(p.detach(), kept[name].detach()) and not torch.equal(p.detach(), fresh[name].detach())
    trainer = tr.build_trainer(branch, model, proc, smoke_dataset(16), {}, runs / "BRANCH",
                               continue_parent_stream=False)  # stream skip: tests/test_opus_g.py
    assert trainer.optimizer.state, "optimizer state was not loaded from keep/optimizer.pt"
    assert trainer.lr_scheduler.get_last_lr()[0] == pytest.approx(1e-3)
    tr.train(trainer, branch, runs / "BRANCH")
    assert trainer.state.global_step == 2


def test_smoke_dataset_shape():
    ds = smoke_dataset(64)
    assert len(ds) == 64 and set(ds.column_names) == {"prompt", "completion"}
