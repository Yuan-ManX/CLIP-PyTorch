import gzip
import html
import os
from functools import lru_cache
import ftfy
import regex as re


@lru_cache()
def default_bpe():
    """
    返回默认的 BPE（字节对编码）词汇表文件的路径。

    BPE 词汇表文件用于将文本拆分为子词单元，从而提高模型的词汇覆盖率。

    返回:
        str: BPE 词汇表文件的绝对路径。
    """
    # 获取当前脚本文件所在的目录
    # 构建 BPE 词汇表文件的完整路径
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    生成一个字典，将 UTF-8 字节映射到对应的 Unicode 字符串。

    可逆的 BPE 编码在 Unicode 字符串上工作。为了避免在词汇表中出现未知的字符（UNK），需要大量的 Unicode 字符。
    当处理大约 100 亿个标记的数据集时，为了获得良好的覆盖率，通常需要大约 5000 个字符。
    这是一个相当大的比例，相对于通常的 32K 大小的 BPE 词汇表。
    为了避免这种情况，我们希望建立 UTF-8 字节和 Unicode 字符串之间的查找表。
    同时避免映射到 BPE 代码无法处理的空白字符和控制字符。

    返回:
        dict: 一个字典，将 UTF-8 字节（整数）映射到对应的 Unicode 字符串。
    """
    # 生成一个包含从 '!' 到 '~' 的所有可打印 ASCII 字符的列表
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    # 初始化 cs 列表为 bs 的副本
    cs = bs[:]
    n = 0
    # 遍历所有可能的 8 位字节值
    for b in range(2**8):
        if b not in bs:
            # 如果当前字节不在 bs 中，则将其添加到 bs 中
            bs.append(b)
            # 并在 cs 中添加一个对应的 Unicode 字符，从 256 开始递增
            cs.append(2**8+n)
            n += 1
    # 将 cs 中的整数转换为对应的 Unicode 字符
    cs = [chr(n) for n in cs]
    # 构建一个字典，将 bs 和 cs 中的元素配对
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    返回一个单词中所有的符号对。

    单词被表示为一个符号元组（符号可以是可变长度的字符串）。

    参数:
        word (tuple): 一个包含符号的元组。

    返回:
        set: 一个包含所有符号对的集合。
    """
    # 初始化一个空集合，用于存储符号对
    pairs = set()
    # 获取第一个符号作为前一个符号
    prev_char = word[0]
    # 遍历剩余的符号
    for char in word[1:]:
        # 添加符号对到集合中
        pairs.add((prev_char, char))
        # 更新前一个符号
        prev_char = char
    # 返回符号对集合
    return pairs


def basic_clean(text):
    """
    对输入文本进行基本的清理。

    清理步骤包括：
    1. 使用 ftfy.fix_text 修复常见的文本编码问题。
    2. 使用 html.unescape 解码 HTML 实体两次，以确保所有实体都被正确解码。
    3. 去除文本开头和结尾的空白字符。

    参数:
        text (str): 需要清理的输入文本。

    返回:
        str: 清理后的文本。
    """
    # 使用 ftfy.fix_text 修复常见的文本编码问题
    text = ftfy.fix_text(text)
    # 使用 html.unescape 解码 HTML 实体两次，以确保所有实体都被正确解码
    text = html.unescape(html.unescape(text))
    # 去除文本开头和结尾的空白字符
    return text.strip()


def whitespace_clean(text):
    """
    对输入文本进行空白字符清理。

    清理步骤包括：
    1. 使用正则表达式将所有连续的空白字符（包括空格、制表符、换行符等）替换为一个空格。
    2. 去除文本开头和结尾的空白字符。

    参数:
        text (str): 需要清理的输入文本。

    返回:
        str: 清理后的文本。
    """
    # 使用正则表达式将所有连续的空白字符替换为一个空格
    text = re.sub(r'\s+', ' ', text)
    # 去除文本开头和结尾的空白字符
    text = text.strip()
    return text


class SimpleTokenizer(object):
    """
    一个简单的分词器，实现了 BPE（字节对编码）分词功能。

    该分词器将文本转换为子词单元，并通过 BPE 算法进行编码和解码。
    """
    def __init__(self, bpe_path: str = default_bpe()):
        """
        初始化 SimpleTokenizer。

        参数:
            bpe_path (str, 可选): BPE 词汇表文件的路径。默认为 `default_bpe()` 函数返回的路径。
        """
        # 获取字节到 Unicode 的映射字典
        self.byte_encoder = bytes_to_unicode()
        # 反转字节到 Unicode 的映射字典，得到 Unicode 到字节的映射
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # 读取 BPE 词汇表文件
        # BPE 词汇表文件是一个 gzip 压缩的文本文件，包含 BPE 合并规则
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        # 跳过前几行（具体跳过行数根据实际情况调整），这里假设跳过 1 到 49152-256-2+1 行
        merges = merges[1:49152-256-2+1]
        # 将每一行的合并规则拆分为元组形式
        merges = [tuple(merge.split()) for merge in merges]

        # 初始化词汇表，包含所有单字节的 Unicode 字符
        vocab = list(bytes_to_unicode().values())
        # 为每个单字节字符添加 '</w>'，表示词尾
        vocab = vocab + [v+'</w>' for v in vocab]
        # 将所有 BPE 合并后的符号添加到词汇表中
        for merge in merges:
            vocab.append(''.join(merge))

        # 添加特殊的开始和结束标记
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        # 构建编码器字典，将词汇表中的每个符号映射到一个唯一的整数索引
        self.encoder = dict(zip(vocab, range(len(vocab))))
        # 构建解码器字典，反转编码器字典，实现整数索引到符号的映射
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 构建 BPE 合并等级字典，将每个合并规则映射到其优先级（索引）
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # 初始化缓存字典，用于存储已经计算过的 BPE 结果
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        # 定义正则表达式模式，用于匹配各种类型的符号
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        """
        应用 BPE 算法对单个标记进行编码。

        参数:
            token (str): 需要进行 BPE 编码的标记。

        返回:
            str: BPE 编码后的标记。
        """
        # 如果标记已经在缓存中，则直接返回缓存的结果
        if token in self.cache:
            return self.cache[token]
        # 对标记进行预处理，添加词尾标记 '</w>'
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        # 获取标记中的所有符号对
        pairs = get_pairs(word)

        if not pairs:
            # 如果没有符号对，则直接返回添加了词尾标记的标记
            return token+'</w>'

        # 循环进行 BPE 合并
        while True:
            # 找到优先级最高的符号对
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                # 如果符号对不在 BPE 合并等级字典中，则停止合并
                break
            # 获取符号对中的两个符号
            first, second = bigram
            new_word = []
            i = 0
            # 遍历单词中的符号，进行合并
            while i < len(word):
                try:
                    # 找到当前符号对在单词中的位置
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    # 如果没有找到，则将剩余的符号添加到新单词中
                    new_word.extend(word[i:])
                    break
                
                # 如果当前符号对匹配，则合并符号
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # 更新单词为合并后的新单词
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                # 如果单词长度为1，则停止合并
                break
            else:
                # 否则，更新符号对
                pairs = get_pairs(word)

        # 将单词中的符号用空格连接
        word = ' '.join(word)
        # 将结果存入缓存
        self.cache[token] = word
        return word

    def encode(self, text):
        """
        对输入文本进行编码。

        编码步骤包括：
        1. 清理文本。
        2. 对每个标记应用 BPE 编码。
        3. 将标记转换为对应的整数索引。

        参数:
            text (str): 需要编码的输入文本。

        返回:
            List[int]: 编码后的整数索引列表。
        """
        bpe_tokens = []
        # 清理文本：去除空白字符，修复编码问题，解码 HTML 实体，并转换为小写
        text = whitespace_clean(basic_clean(text)).lower()
        # 使用正则表达式匹配所有标记
        for token in re.findall(self.pat, text):
            # 将每个标记编码为字节，并转换为对应的 Unicode 字符串
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # 对每个 BPE 标记进行编码，并添加到结果列表中
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        """
        对编码后的整数索引列表进行解码。

        解码步骤包括：
        1. 将整数索引转换为对应的符号。
        2. 将符号连接成文本。
        3. 替换词尾标记 '</w>' 为空格。

        参数:
            tokens (List[int]): 需要解码的整数索引列表。

        返回:
            str: 解码后的文本。
        """
        # 将整数索引转换为对应的符号，并连接成字符串
        text = ''.join([self.decoder[token] for token in tokens])
        # 将字节字符串解码为 UTF-8 字符串，并替换词尾标记 '</w>' 为空格
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
