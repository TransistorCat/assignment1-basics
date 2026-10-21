
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
    vocab = {}
    merges = []
    return vocab, merges

test=[[1, 2], [3, 4]]

if __name__ == "__main__":
    vocab, merges = train_bpe("/home/user/learn/assignment1-basics/tests/fixtures/tinystories_sample.txt", 100, ["<|endoftext|>"])
    print(vocab)
    print(merges)

