import torch
from performer_pytorch import PerformerLM


def get_performer(n_positions=512, n_head=12, n_layer=12, vocab_size=77, n_embd=768,
                  local_window_size=50, local_attn_heads=6,
                  feature_redraw=100, generalized_attention=True):
    return PerformerLM(
        num_tokens=vocab_size,
        max_seq_len=n_positions,  # max sequence length
        dim=n_embd,  # dimension
        depth=n_layer,  # layers
        heads=n_head,  # heads
        causal=True,  # auto-regressive or not
        # nb_features=256,
        # number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head
        feature_redraw_interval=feature_redraw,
        # how frequently to redraw the projection matrix, the more frequent, the slower the training
        generalized_attention=generalized_attention,
        # defaults to softmax approximation, but can be set to True for generalized attention
        kernel_fn=torch.nn.ReLU(),
        # the kernel function to be used, if generalized attention is turned on, defaults to Relu
        reversible=True,  # reversible layers, from Reformer paper
        ff_chunks=10,  # chunk feedforward layer, from Reformer paper
        use_scalenorm=False,  # use scale norm, from 'Transformers without Tears' paper
        use_rezero=False,  # use rezero, from 'Rezero is all you need' paper
        ff_glu=False,  # use GLU variant for feedforward
        emb_dropout=0.1,  # embedding dropout
        ff_dropout=0.1,  # feedforward dropout
        attn_dropout=0.1,  # post-attn dropout
        local_attn_heads=local_attn_heads,
        local_window_size=local_window_size  # window size of local attention
    )

