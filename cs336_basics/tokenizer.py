
import regex as re
import collections


def train_bpe(input_path:str,vocab_size:int,special_tokens:list[str]) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    """
    Train a BPE tokenizer on the input text.
    """

    if special_tokens is None:
        special_tokens = []
    if "" in special_tokens:
        raise ValueError("'' is a special token")
    with open(input_path, 'rb') as f:
         raw = f.read()
    raw = raw.decode('utf-8')
    chunks=[raw]
    for special_token in special_tokens :
        new_chunks = []
        for i in chunks:
            for j in i.split(special_token):
                new_chunks.append(j)
        chunks = new_chunks

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_counts = collections.Counter()
    for chunk in chunks:
        tokens = re.findall(PAT, chunk)
        word_counts.update(tokens)
    words: dict[bytes, tuple[int, list[int]]] = {}
    for word, count in word_counts.items():
        word = word.encode('utf-8')
        words[word] = (count, list(word))

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    pair_counts: dict[tuple[int,int], int] = {}
    pair_index: dict[tuple[int,int], set] = {}
    for word, (count, ids) in words.items():
        pairs = get_pairs(ids)
        for pair in pairs:
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            pair_index.setdefault(pair, set()).add(word)
    
    num_merges = vocab_size - 256 - len(special_tokens)
    merges: list[tuple[bytes,bytes]] = []
    if num_merges < 0:
        raise ValueError("vocab_size is too small")
    for i in range(num_merges):
        best_count = -1
        best_pair = None
        for pair, count in pair_counts.items():
            if best_count < count :
                best_count = count
                best_pair = pair
            if best_count == count and(vocab[pair[0]], vocab[pair[1]]) > (vocab[best_pair[0]], vocab[best_pair[1]]):
                best_pair = pair
        if best_pair is None:
            break
        new_pair_id = len(vocab)
        vocab[new_pair_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        need_merge_words = pair_index[best_pair].copy()

        for word in need_merge_words:
            count,old_ids = words[word]
            old_pairs = get_pairs(old_ids)
            new_ids = merge_word_pair(old_ids, best_pair, new_pair_id)
            new_pairs = get_pairs(new_ids)
            words[word] = (count, new_ids)

            for pair in old_pairs:
                pair_counts[pair] -= count
                pair_index.setdefault(pair, set()).discard(word)
            for pair in new_pairs:
                pair_counts[pair] = pair_counts.get(pair, 0) + count
                pair_index.setdefault(pair, set()).add(word)
        for pair, count in pair_counts.copy().items():
            if count <= 0:
                pair_counts.pop(pair, None)
                pair_index.pop(pair, None)
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
    return vocab, merges

def get_pairs(ids: list[int]) -> list[tuple[int, int]]:
    """
    Get all pairs of adjacent ids in the list.
    """
    pairs = []
    for i in range(len(ids) - 1):
        pairs.append((ids[i], ids[i + 1]))
    return pairs


def merge_word_pair(ids: list[int], pair: tuple[int, int],new_pair_id: int) -> list[int]:
    """
    Merge a pair of ids into a new id.
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_pair_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


if __name__ == "__main__":
    vocab, merges = train_bpe("/home/user/learn/assignment1-basics/tests/fixtures/tinystories_sample.txt", 400, ["<|endoftext|>"])
    print(vocab)
    print(merges)

    # merge_word_pair([1, 2, 3, 4, 5], (2, 3), 6)
    # print(merge_word_pair([1, 3, 4], (1, 2), 99))   # 期望 [1, 3, 4]
    # print(merge_word_pair([116], (1, 2), 99))        # 期望 [116]（单元素词，fixture 里真实存在）