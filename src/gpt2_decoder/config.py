from dataclasses import dataclass
import json


@dataclass(frozen=True)
class GPT2Config:
    vocab_size: int
    n_positions: int
    n_ctx: int
    n_embd: int
    n_layer: int
    n_head: int
    activation_function: str
    layer_norm_epsilon: float
    bos_token_id: int
    eos_token_id: int


    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def intermadiate_size(self) -> int:
        return 4 * self.n_embd


    @classmethod
    def from_json(cls, path: str) -> "GPT2Config":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        config = cls(
            vocab_size=cfg["vocab_size"],
            n_positions=cfg["n_positions"],
            n_ctx=cfg.get("n_ctx", cfg["n_positions"]),
            n_embd=cfg["n_embd"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            activation_function=cfg.get("activation_function"),
            layer_norm_epsilon=cfg.get("layer_norm_epsilon"),
            bos_token_id=cfg.get("bos_token_id"),
            eos_token_id=cfg.get("eos_token_id"),
        )

        assert config.n_embd % config.n_head == 0

        return config