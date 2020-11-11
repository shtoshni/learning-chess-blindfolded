import torch


def get_strided_attn_mask(stride_size, max_seq_length=800):
    assert (stride_size > 0)
    counter = 0
    causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length))
    for i in range(max_seq_length):
        block_i = i // stride_size
        for j in range(i + 1):
            block_j = j // stride_size
            j_seq_idx = j % stride_size - stride_size
            if block_i == block_j:
                causal_mask[i, j] = 1.0
            elif (block_i == block_j + 1) and (j_seq_idx == -1):
                # Last token of the last block
                causal_mask[i, j] = 1.0
            else:
                causal_mask[i, j] = 0
                counter += 1

    return causal_mask.view(1, 1, max_seq_length, max_seq_length)


def get_last_window_attn_mask(window_size, max_seq_length=800):
    assert (window_size > 0)
    counter = 0
    causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length))
    for i in range(max_seq_length):
        for j in range(i + 1):
            if (i - j) <= window_size:
                causal_mask[i, j] = 1.0
            else:
                causal_mask[i, j] = 0
                counter += 1

    return causal_mask.view(1, 1, max_seq_length, max_seq_length)


if __name__ == '__main__':
    # print(get_strided_attn_mask(10, max_seq_length=800))
    print(get_last_window_attn_mask(5, max_seq_length=20))




