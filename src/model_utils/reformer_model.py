from transformers import ReformerModelWithLMHead, ReformerConfig


def get_reformer(vocab_size=77, n_layer=12, n_embd=768, n_head=12,  n_positions=512, local_window_size=50,
                 num_buckets=8, num_hashes=1, hash_seed=None):
    attn_layers = ["local", "local", "lsh", "local", "local", "local", "lsh", "local",
                   "local", "local", "lsh", "local"]
    # attn_layers = ["local", "lsh", "local", "lsh", "local", "lsh", "local", "lsh", "local", "lsh", "local", "lsh"]
    config = ReformerConfig(
        hash_seed=hash_seed,
        attn_layers=attn_layers[:n_layer],
        hidden_size=n_embd,
        max_position_embeddings=n_positions,
        feed_forward_size=3072,
        vocab_size=vocab_size,
        is_decoder=True,
        axial_pos_embds_dim=[256, 512],
        axial_pos_shape=[14, 25],
        num_hashes=num_hashes,
        local_attn_chunk_length=local_window_size,
        num_buckets=num_buckets,
        lsh_attn_chunk_length=local_window_size,
        num_attention_heads=n_head,
    )
    return ReformerModelWithLMHead(config=config)

