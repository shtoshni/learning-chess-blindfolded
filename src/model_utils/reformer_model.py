from transformers import ReformerModelWithLMHead, ReformerConfig


def get_reformer(vocab_size=77, n_embd=768, num_buckets=32, n_layer=12, n_positions=512, num_hashes=1):
    attn_layers = ["local", "local", "lsh", "local", "local", "local", "lsh", "local",
                   "local", "local", "lsh", "local"]
    config = ReformerConfig(
        attn_layers=attn_layers[:n_layer],
        hidden_size=n_embd,
        max_position_embeddings=n_positions,
        feed_forward_size=3072,
        vocab_size=vocab_size,
        is_decoder=True,
        axial_pos_embds_dim=[256, 512],
        axial_pos_shape=[16, 24],
        num_buckets=num_buckets,
        num_hashes=num_hashes,
    )
    return ReformerModelWithLMHead(config=config)

