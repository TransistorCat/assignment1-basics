
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
    print(raw[:100])
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
    print(word_counts)
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
            if best_count == count and pair > best_pair:
                best_pair = pair
        if best_pair is None:
            break
        vocab[len(vocab)] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))


    return vocab, merges



def get_pairs(ids: list[int]) -> list[tuple[int, int]]:
    """
    Get all pairs of adjacent ids in the list.
    """
    pairs = []
    for i in range(len(ids) - 1):
        pairs.append((ids[i], ids[i + 1]))
    return pairs


def merge_word_pair(ids: int, pair: tuple[int, int],new_pair_id: int) -> bytes:
    """
    Merge a pair of ids into a new id.
    """

def test_get_pairs():
    ids = [1, 2, 3, 4, 5]
    pairs = get_pairs(ids)
    print(pairs)

if __name__ == "__main__":
    vocab, merges = train_bpe("/home/user/learn/assignment1-basics/tests/fixtures/tinystories_sample.txt", 400, ["<|endoftext|>"])
    print(vocab)
    print(merges)

