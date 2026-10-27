
from typing import Iterable, Iterator
import regex as re
import collections

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
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
    chunks = split_on_specials(raw, special_tokens)
    word_counts = collections.Counter()
    for chunk in chunks:
        if chunk in special_tokens:
            continue
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

def apply_merge(pieces:list[bytes], pair:tuple[bytes,bytes]) -> list[bytes]:
    new_pieces = []
    i = 0
    while i < len(pieces):
        if i < len(pieces) - 1 and pieces[i] == pair[0] and pieces[i + 1] == pair[1]:
            new_pieces.append(pair[0] + pair[1])
            i += 2
        else:
            new_pieces.append(pieces[i])
            i += 1
    return new_pieces

def find_best_pair(pieces:list[bytes], merge_ranks:dict[tuple[bytes,bytes], int]) -> tuple[bytes,bytes] | None:
    best_pair = None
    best_rank = float('inf')
    for pair in get_pairs(pieces):
        rank = merge_ranks.get(pair, float('inf'))
        if rank < best_rank:
            best_rank = rank
            best_pair = pair
    return best_pair

def split_on_specials(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split the text on special tokens.
    """

    chunks=[text]
    for special_token in sorted(special_tokens, key=len, reverse=True):
        new_chunks = []
        for i in chunks:
            if i in special_tokens:
                new_chunks.append(i)
                continue
            for j in  re.split("("+re.escape(special_token)+")", i):
                new_chunks.append(j)
        chunks = new_chunks
    return chunks

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes,bytes]], special_tokens: list[str]):
        self.vocab = vocab
        self.merges = merges
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = special_tokens
        # self.vocab.update({len(vocab) + i: token.encode('utf-8') for i, token in enumerate(special_tokens)})
        if self.special_tokens is None:
            self.special_tokens = []
        
        if "" in self.special_tokens:
            raise ValueError("'' is a special token")
        
        for token in self.special_tokens:
            bytes_token = token.encode('utf-8')
            if bytes_token not in self.vocab.values():
                self.vocab[len(self.vocab)] = bytes_token


        self.inverse_vocab= {v: k for k, v in self.vocab.items()}
        self.special_tokens_ids = {token:self.inverse_vocab[token.encode('utf-8')] for token in self.special_tokens}


    
    def from_files(cls,vocab_filepath:str, merges_filepath:str, special_tokens:list[str]):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        ids = []
        chunks = split_on_specials(text, self.special_tokens)
        for chunk in chunks:
            if chunk in self.special_tokens:
                ids.append(self.special_tokens_ids[chunk])
                continue
            words = re.findall(PAT, chunk)
            for word in words:
                pieces = [bytes([byte]) for byte in word.encode('utf-8')]
                while best_pair := find_best_pair(pieces, self.merge_ranks):
                    pieces = apply_merge(pieces, best_pair)
                ids.extend(self.inverse_vocab[piece] for piece in pieces)
        return ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    def decode(self, tokens: list[int]) -> str:
        text = []
        for token in tokens:
            text.append(self.vocab[token])
        bytes_text = b''.join(text)
        return bytes_text.decode('utf-8',errors="replace")


if __name__ == "__main__":
    # vocab, merges = train_bpe("/home/user/learn/assignment1-basics/tests/fixtures/tinystories_sample.txt", 400, ["<|endoftext|>"])
    # print(vocab)
    # print(merges)
    # print(re.escape("<eot>"))                                    # ① 疫苗长什么样
    # print("(" + re.escape("<eot>") + ")")                        # ② 最终模式字符串
    # print(re.split(re.escape("<eot>"), "hi<eot>yo")) # ③ 用它切分文本
    # print(split_on_specials("hi<eot>yo", ["<eot>"]))
    # tok = Tokenizer({0: b"a", 1: b"b"}, [], ["<eot>"])
    # print(tok.encode("a b"))
    # print(apply_merge([b'a', b'b', b'a', b'b'], (b'a', b'b')))   # [b'ab', b'ab']
    # print(apply_merge([b'a', b'b', b'c'], (b'b', b'c')))          # [b'a', b'bc']
    # print(apply_merge([b'x'], (b'a', b'b')))                      # [b'x']
    # merge_word_pair([1, 2, 3, 4, 5], (2, 3), 6)
    # print(merge_word_pair([1, 3, 4], (1, 2), 99))   # 期望 [1, 3, 4]
    # print(merge_word_pair([116], (1, 2), 99))        # 期望 [116]（单元素词，fixture 里真实存在）
    # vocab = {0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t",
    #      6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}
    # merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]
    # tok = Tokenizer(vocab, merges, ["<|endoftext|>", "<|endoftext|><|endoftext|>"])
    # print(tok.encode("the cat ate"))  

    # text = "the cat ate the cat ate"
    # tok = Tokenizer({0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t",
    #      6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}, [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")], [])
    # print(list(tok.encode_iterable(["the cat", " ate", " the cat ate"])))
    # print(tok.encode(text))

    # from tests.test_tokenizer import VOCAB_PATH, MERGES_PATH, get_tokenizer_from_vocab_merges_path

    # tok = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"])

    # test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    # b = tok.decode(tok.encode(test_string))

    # print(len(test_string), len(b))                     # 长度差 = 丢了/多了几个字符
    # for i, (x, y) in enumerate(zip(test_string, b)):
    #     if x != y:
    #         print("第一个分歧点在索引", i)
    #         print("期望:", repr(test_string[max(0, i-10):i+10]))    # 分歧点上下文各 10 字符
    #         print("实际:", repr(b[max(0, i-10):i+10]))
    #         break

    ranks = {(b'a', b'b'): 3, (b'ab', b'c'): 5}
    print(find_best_pair([b'a', b'b', b'a', b'b'], ranks))   # (b'a', b'b')
    print(find_best_pair([b'a', b'b', b'c'], ranks))          # (b'a', b'b')
    print(find_best_pair([b'x'], ranks))                      # None
    print(find_best_pair([b'q', b'r'], ranks))                # None