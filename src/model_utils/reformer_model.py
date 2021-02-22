from transformers import ReformerModelWithLMHead, ReformerConfig


def get_reformer(vocab_size=77, n_layer=12, n_embd=768, n_head=12,  n_positions=512, local_window_size=50,
                 num_buckets=None, num_hashes=1):
    attn_layers = ["local", "local", "lsh", "local", "local", "local", "lsh", "local",
                   "local", "local", "lsh", "local"]
    # attn_layers = ["local", "lsh", "local", "lsh", "local", "lsh", "local", "lsh", "local", "lsh", "local", "lsh"]
    config = ReformerConfig(
        hash_seed=None,
        attn_layers=attn_layers[:n_layer],
        # attention_head_size=128,
        hidden_size=n_embd,
        max_position_embeddings=350,
        feed_forward_size=3072,
        vocab_size=vocab_size,
        is_decoder=True,
        axial_pos_embds_dim=[256, 512],
        axial_pos_shape=[14, 25],
        num_hashes=num_hashes,
        num_buckets=num_buckets,
        local_attn_chunk_length=local_window_size,
        # num_buckets=num_buckets,
        lsh_attn_chunk_length=local_window_size,
        num_attention_heads=n_head,
        # lsh_attention_probs_dropout_prob=0.1,
        # local_attention_probs_dropout_prob=0.1,
        # hidden_dropout_prob=0.1,
        chunk_size_feed_forward=0,
        chunk_size_lm_head=0,
        eos_token_id=2,
        hidden_act='relu',
    )
    return ReformerModelWithLMHead(config=config)

