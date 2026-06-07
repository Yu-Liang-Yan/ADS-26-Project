"""
Phase 3：Prefix Cache

在 Phase 2 里，KV Cache 解决了**单次请求内部**「每 decode 一步都要重算整段历史」
的问题。但当我们面对**多次请求**时，仍然有大量重复计算：例如同一个系统提示词会
在每次对话里重复，多轮对话里每一轮都要对「之前的所有轮次」重新 prefill 一遍。

Prefix Cache 的核心想法：把过去请求 prefill 结束时的缓存快照**跨请求**保存下来；
当新请求到来时，如果它的 prompt 以某个已缓存 prompt 作为前缀，就直接加载那份快照，
只对「新 prompt 相对老 prompt 多出来的后缀」做 prefill —— 前缀部分的计算**彻底
省掉**。

Phase 3 实现的是**整 prompt 粒度**的Prefix Cache（最简单、最常用的一种形式）：
  - 每次 prefill 结束，将 (prompt_token_ids, 缓存快照) 作为一条记录存入 PrefixCache。
  - 新请求来时，在已保存的记录里找「最长的、作为新 prompt 前缀的」那条记录，
    加载其缓存，把新 prompt 去掉已匹配前缀后的剩余 token 交给 forward。
  - 由于被复用的缓存会被后续请求继续写入，存取时都必须 clone 以避免互相污染。

淘汰策略：**LRU（Least Recently Used）**
----------------------------------------
容量有限时需要选择丢弃哪条记录。朴素的 FIFO 会把"最早插入"的记录丢掉，但
"最早"未必"最没用"——一条很早就插入、但每次请求都会命中的系统提示词，FIFO
同样会把它淘汰。LRU 改成以"最近一次被使用的时间"排序：每当一条记录被命中
（lookup）或被再次写入（insert 遇到相同 key）时，把它标记为"最近使用"；容量
溢出时丢弃"最久未使用"的那一条。这样长期被频繁命中的热点前缀会一直留在缓存里。

实现上用 `collections.OrderedDict` 维护"按最近使用时间排序"的记录：
  - 队尾（last）= 最近使用
  - 队首（first）= 最久未使用
  - 命中或复写 → `move_to_end(key)`
  - 容量溢出 → `popitem(last=False)`
"""
from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cache import Qwen3_5DynamicCache


class PrefixCache:
    """
    最简单的Prefix Cache实现：用一个 OrderedDict 保存所有「prompt → cache 快照」
    记录，lookup 时线性扫描找最长前缀匹配，命中后把该条目移到队尾以更新 LRU 序。
    对于教学与小规模演示足够直观。真实推理引擎（如 vLLM）会用 radix tree + 块级
    哈希来支持 O(log n) 匹配与更细粒度的块级共享，核心思想是一致的。
    """

    def __init__(self, max_entries: int = 32):
        # OrderedDict 从队首到队尾按「最久未使用 → 最近使用」排列。
        # key 是 prompt_token_ids 的 tuple；value 是 cache 快照。
        self._entries: "OrderedDict[tuple[int, ...], Qwen3_5DynamicCache]" = OrderedDict()
        self._max_entries = max_entries

        # 统计信息（测试脚本会读取这两个计数器以展示命中率）
        self.hits: int = 0
        self.misses: int = 0
        self.hit_tokens: int = 0  # 累计被Prefix Cache命中而省下的 prefill token 数
        self.evictions: int = 0   # 累计因容量溢出被 LRU 淘汰的条目数

    # ---------- 查询 ----------

    def lookup(
        self, token_ids: list[int]
    ) -> tuple[int, "Qwen3_5DynamicCache | None"]:
        """
        在已保存的记录里找「最长的、作为 token_ids 前缀的」一条，返回
        (matched_len, cloned_cache)。

        约束（为简化实现与避免 forward 收到空输入）
        --------------------------------------------
        1. 若没有任何记录满足「该条记录的 tokens 是 token_ids 的前缀」，返回 (0, None)。
        2. 若找到的 matched_len == len(token_ids)（新 prompt 和某条记录完全相等），
           则把 matched_len 截断为 len(token_ids) - 1 —— forward 至少需要 1 个新 token
           来产生本次请求的首个输出 logits。
        3. 命中时，为避免快照被后续请求污染，**必须返回 cache 的 clone 而非原对象**。
        4. **LRU 维护**：命中后必须把命中的那条记录 `move_to_end`，把它标记为「最近
           使用」，这样后续 insert 触发淘汰时不会被丢掉。

        实现提示
        --------
        1. 维护一个 best_key = None, best_len = 0，遍历 self._entries.items()，
           对每一条 (cached_tokens, cached_cache)：
             - 用 tuple 切片判断 cached_tokens 是否等于 token_ids 的前 len(cached_tokens) 项；
             - 若是，则它是一个合法前缀；比较长度，择最长者。
        2. 未命中：self.misses += 1，返回 (0, None)。
        3. 命中：
             - 按约束 2 处理完全匹配的情况；
             - self._entries.move_to_end(best_key) 更新 LRU 顺序；
             - self.hits += 1, self.hit_tokens += matched_len；
             - 返回 (matched_len, self._entries[best_key].clone())。
        """
        # ===== TODO: Prefix Cache - (START) =====
        best_key = None
        best_len = 0
        token_tuple = tuple(token_ids)

        for cached_tokens, _ in self._entries.items():
            if (
                len(cached_tokens) <= len(token_ids)
                and cached_tokens == token_tuple[: len(cached_tokens)]
            ):
                if len(cached_tokens) > best_len:
                    best_len = len(cached_tokens)
                    best_key = cached_tokens

        if best_key is None:
            self.misses += 1
            return 0, None

        matched_len = best_len
        if matched_len == len(token_ids):
            matched_len = len(token_ids) - 1

        self._entries.move_to_end(best_key)
        self.hits += 1
        self.hit_tokens += matched_len
        return matched_len, self._entries[best_key].clone()
        # ===== TODO: Prefix Cache - (END) =====

    # ---------- 插入 ----------

    def insert(
        self, token_ids: list[int], cache: "Qwen3_5DynamicCache"
    ) -> None:
        """
        在一次 prefill 完成后，把 (prompt tokens, 当前缓存的深拷贝) 追加到记录中。

        约束
        ----
        1. 若已存在一条记录其 tokens 与 token_ids 完全相等，**不重新 clone**，但
           要把它 `move_to_end` 以更新 LRU 顺序（这条前缀刚刚又被用过一次）。
        2. 若 self._entries 已达 self._max_entries，按 LRU 淘汰"最久未使用"的一条：
           `self._entries.popitem(last=False)`，并把 self.evictions 加 1。
        3. 存入的必须是 cache.clone() —— 否则后续 decode 向该 cache 追加 K/V 时
           会把本快照一起"带着"往前走。

        实现提示
        --------
        1. 先把 token_ids 转成 tuple（list 不可 hash、也不便作键）。
        2. 若 key 已在 self._entries 中：move_to_end(key) 后 return。
        3. 若 len(self._entries) >= self._max_entries：popitem(last=False)，
           self.evictions += 1。
        4. self._entries[key] = cache.clone()（新插入的条目天然位于队尾 = 最近使用）。
        """
        # ===== TODO: Prefix Cache - (START) =====
        key = tuple(token_ids)
        if key in self._entries:
            self._entries.move_to_end(key)
            return

        if len(self._entries) >= self._max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1

        self._entries[key] = cache.clone()
        # ===== TODO: Prefix Cache - (END) =====

    # ---------- 辅助 ----------

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """重置所有记录与统计信息。"""
        self._entries.clear()
        self.hits = 0
        self.misses = 0
        self.hit_tokens = 0
        self.evictions = 0
