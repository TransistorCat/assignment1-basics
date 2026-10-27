# Assignment 1 Report

## Unicode
### unicode1: Understanding Unicode (1 pt)
>>> chr(0)
'\x00'
>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
是空字符；__repr__ 写成 \x00 而 print 没有字形；它还在字符串里，但 C 风格接口会把它当成结尾。
### unicode2: Unicode Encodings (3 pts)
>>> "hello".encode("utf-8")
b'hello'
>>> len("hello".encode("utf-8"))
5
>>> "hello".encode("utf-16")
b'\xff\xfeh\x00e\x00l\x00l\x00o\x00'
>>> len("hello".encode("utf-16"))
12
>>> "hello".encode("utf-32")
b'\xff\xfe\x00\x00h\x00\x00\x00e\x00\x00\x00l\x00\x00\x00l\x00\x00\x00o\x00\x00\x00'
>>> len("hello".encode("utf-32"))
24
>>> "牛".encode("utf-8")
b'\xe7\x89\x9b'
>>> len("牛".encode("utf-8"))
3
>>> "牛".encode("utf-16")
b'\xff\xfe[r'
>>> len("牛".encode("utf-16"))
4
>>> "牛".encode("utf-32")
b'\xff\xfe\x00\x00[r\x00\x00'
>>> len("牛".encode("utf-32"))
8
英文在 UTF-8 里一字节一个字符，UTF-16 和 UTF-32 会插入 \x00 和 BOM。
因为 UTF-8 中一个字符可能需要多个字节，不能把每个字节都当作完整字符单独解码
对 b'\xe7\x89\x9b'（「牛」的 UTF-8）会报错，因为函数把多字节字符拆开逐字节解码了

两字节的合法 UTF-8 长这样：第一个字节在 0xC2–0xDF（二进制以 110 开头），第二个字节在 0x80–0xBF（以 10 开头）。两个条件都要满足。
195（0xC3）是合法的两字节引导字节，它要求后面跟一个延续字节。延续字节的范围是 0x80–0xBF。119（0x77）是字母 w，落在这个范围外面，所以 bytes([195, 119]).decode("utf-8") 会报 UnicodeDecodeError，对不上任何字符
## train_bpe_tinystories (2 pts)
### (a) 训练耗时/内存 + 最长 token 是否合理
训练在 2.2GB TinyStories 上完成（10,000 词表），总耗时 49.7 秒——其中预处理 35.1 秒（find_chunk_boundaries 切块 + 8 进程并行正则切词）、合并循环 14.6 秒；峰值内存主进程 0.09GB、worker 最高 1.70GB，远在 30 分钟 / 30GB 限制之内。最长的 token 是 b' accomplishment'（15 字节）：GPT-2 预切分把空格并入词首，加上 TinyStories 语言简单重复，高频完整英文单词自然被逐轮合并为单一 token——符合 BPE“给常见词发整票”的设计意图

### (b) 哪步最耗时
35.1s/49.7s ≈ 70% 在预处理，正则切 2.2GB 是大头，8 进程已压制

## train_bpe_expts_owt (2 pts)
### (a) 耗时/内存/最长 token
预处理 276.6s、合并循环 ~70 分钟、对表峰值 ~400 万

### (b) TS vs OWT 两个 tokenizer 对比
初始对表 2,108 vs 19,592、峰值 4.7 万 vs ~400 万、两边最长 token  accomplishment'（15 字节）和 64 字节的 Ã 重复序列 🤯
64 字节的 Ã 重复是网页里双重 UTF-8 误读留下的，BPE 把它当成高频字节序列合并了

## tokenizer_experiments (4 pts)
### (a) 压缩率（bytes/token）
ts_sum_bytes/ts_sum_tokens:  4.144026186579378
owt_sum_bytes/owt_sum_tokens:  4.643767552387125

### (b) 用 TS tokenizer 编 OWT 文本会怎样
owt_sum_bytes/owt_ts_sum_tokens:  3.406926063871939
词表更小 + 领域对不上，所以更碎，但还没碎成字节

### (c) 吞吐 + Pile 估算
吞吐 0.58 MB/s、Pile 估算 396.4 小时、可并行化压缩

### (d) 为什么 uint16 合适
max id 9999/31999 < 65535 = 2¹⁶−1，恰为最小可容纳类型；5 亿 token 省一半 IO/内存