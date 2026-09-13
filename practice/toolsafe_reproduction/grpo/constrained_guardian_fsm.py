"""Tokenizer-aware finite-state grammar for ToolSafe Guardian responses.

This module contains no model or PyTorch dependency.  It separates sampled
semantic actions from deterministic protocol tokens so the same event trace can
later drive rollout-time and recomputed constrained log probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Optional


class GrammarCompilationError(ValueError):
    """The tokenizer cannot represent the Guardian protocol exactly."""


class ConstraintViolation(ValueError):
    """A sampled token is not allowed by the current FSM state."""


class TokenTrie:
    """Immutable-after-build trie for tokenizer-specific enum spellings."""

    TERMINAL = "__terminal__"

    def __init__(self, choices: dict[Hashable, tuple[int, ...]], *, name: str) -> None:
        if not choices:
            raise GrammarCompilationError(f"{name}: choice trie must not be empty")
        self.name = name
        self._root: dict[Any, Any] = {}
        self._choices = dict(choices)
        for value, token_ids in choices.items():
            if not token_ids:
                raise GrammarCompilationError(f"{name}: choice {value!r} has no tokens")
            node = self._root
            for token_id in token_ids:
                if self.TERMINAL in node:
                    raise GrammarCompilationError(
                        f"{name}: one choice tokenization prefixes another"
                    )
                node = node.setdefault(int(token_id), {})
            if node:
                raise GrammarCompilationError(
                    f"{name}: one choice tokenization prefixes another"
                )
            if self.TERMINAL in node:
                raise GrammarCompilationError(f"{name}: duplicate tokenized choices")
            node[self.TERMINAL] = value

    def _node(self, prefix: tuple[int, ...]) -> dict[Any, Any]:
        node = self._root
        for token_id in prefix:
            child = node.get(int(token_id))
            if not isinstance(child, dict):
                raise ConstraintViolation(
                    f"{self.name}: token prefix {prefix!r} is outside the trie"
                )
            node = child
        return node

    def allowed(self, prefix: tuple[int, ...] = ()) -> frozenset[int]:
        return frozenset(
            int(token_id)
            for token_id in self._node(prefix)
            if token_id != self.TERMINAL
        )

    def resolved(self, prefix: tuple[int, ...]) -> tuple[bool, Optional[Hashable]]:
        node = self._node(prefix)
        if self.TERMINAL in node:
            return True, node[self.TERMINAL]
        return False, None

    @property
    def choices(self) -> dict[Hashable, tuple[int, ...]]:
        return dict(self._choices)


@dataclass(frozen=True)
class FSMEvent:
    state_before: str
    state_after: str
    sampled_token_id: Optional[int]
    serialized_token_ids: tuple[int, ...]
    deterministic_token_ids: tuple[int, ...]
    semantic: bool
    hidden: bool = False
    forced: bool = False


@dataclass(frozen=True)
class CompiledGuardianGrammar:
    tokenizer: Any
    tokenizer_name: str
    tokenizer_vocab_size: int
    start_tokens: tuple[int, ...]
    after_rationale_tokens: tuple[int, ...]
    after_malicious_tokens: tuple[int, ...]
    after_attacked_tokens: tuple[int, ...]
    after_harmfulness_tokens: tuple[int, ...]
    rationale_end_trie: TokenTrie
    enum_tries: dict[str, TokenTrie]
    safe_rationale_token_ids: frozenset[int]
    min_rationale_content_tokens: int
    max_rationale_tokens: int


def _encode_exact(tokenizer: Any, text: str, *, name: str) -> tuple[int, ...]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except Exception as exc:
        raise GrammarCompilationError(f"{name}: tokenizer encode failed") from exc
    token_ids = tuple(int(token_id) for token_id in encoded)
    if not token_ids:
        raise GrammarCompilationError(f"{name}: tokenization is empty")
    try:
        decoded = tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        decoded = tokenizer.decode(list(token_ids))
    if decoded != text:
        raise GrammarCompilationError(
            f"{name}: tokenizer round-trip differs: expected {text!r}, got {decoded!r}"
        )
    return token_ids


def _vocabulary_ids(tokenizer: Any) -> set[int]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if isinstance(vocab, dict):
            return {int(token_id) for token_id in vocab.values()}
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        return set(range(vocab_size))
    raise GrammarCompilationError("tokenizer must expose get_vocab() or vocab_size")


def _decode_piece(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([token_id])
    except Exception:
        return ""


def _safe_rationale_vocabulary(tokenizer: Any) -> frozenset[int]:
    special_ids = {
        int(token_id)
        for token_id in (getattr(tokenizer, "all_special_ids", None) or [])
        if token_id is not None
    }
    for name in ("eos_token_id", "pad_token_id", "bos_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int):
            special_ids.add(value)

    allowed: set[int] = set()
    for token_id in _vocabulary_ids(tokenizer):
        if token_id in special_ids:
            continue
        piece = _decode_piece(tokenizer, token_id)
        if not piece or any(character in piece for character in "<>\n\r"):
            continue
        if any(ord(character) < 32 for character in piece):
            continue
        allowed.add(token_id)
    if not allowed:
        raise GrammarCompilationError("safe rationale vocabulary is empty")
    return frozenset(allowed)


def compile_guardian_grammar(
    tokenizer: Any,
    *,
    min_rationale_content_tokens: int = 8,
    max_rationale_tokens: int = 192,
) -> CompiledGuardianGrammar:
    """Compile protocol literals and enum tries for one exact tokenizer."""

    if min_rationale_content_tokens <= 0:
        raise GrammarCompilationError("minimum rationale content must be positive")
    if max_rationale_tokens < min_rationale_content_tokens:
        raise GrammarCompilationError("rationale cap must be >= minimum content")

    start = _encode_exact(tokenizer, "<Think> ", name="Think opening")
    after_rationale = _encode_exact(
        tokenizer,
        " <\\Think>\n<Malicious_User_Request>",
        name="post-rationale protocol",
    )
    after_malicious = _encode_exact(
        tokenizer,
        "<\\Malicious_User_Request>\n<Being_Attacked>",
        name="post-MUR protocol",
    )
    after_attacked = _encode_exact(
        tokenizer,
        "<\\Being_Attacked>\n<Harmfulness_Rating>",
        name="post-attack protocol",
    )
    after_harmfulness = _encode_exact(
        tokenizer,
        "<\\Harmfulness_Rating>",
        name="Harmfulness closing",
    )
    safe_rationale_token_ids = _safe_rationale_vocabulary(tokenizer)
    learned_end_token = after_rationale[0]
    if learned_end_token in safe_rationale_token_ids:
        raise GrammarCompilationError(
            "post-rationale opening token is also legal free text; the tokenizer "
            "cannot provide an unambiguous learned rationale-end action"
        )
    rationale_end = TokenTrie(
        {"end": (learned_end_token,)},
        name="rationale_end",
    )
    enum_tries = {
        "malicious_user_request": TokenTrie(
            {
                False: _encode_exact(tokenizer, " no ", name="MUR no"),
                True: _encode_exact(tokenizer, " yes ", name="MUR yes"),
            },
            name="malicious_user_request",
        ),
        "being_attacked": TokenTrie(
            {
                False: _encode_exact(tokenizer, " no ", name="attack no"),
                True: _encode_exact(tokenizer, " yes ", name="attack yes"),
            },
            name="being_attacked",
        ),
        "harmfulness_rating": TokenTrie(
            {
                0.0: _encode_exact(tokenizer, " 0.0 ", name="harmfulness 0.0"),
                0.5: _encode_exact(tokenizer, " 0.5 ", name="harmfulness 0.5"),
                1.0: _encode_exact(tokenizer, " 1.0 ", name="harmfulness 1.0"),
            },
            name="harmfulness_rating",
        ),
    }
    tokenizer_name = str(getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__))
    vocab_size = int(getattr(tokenizer, "vocab_size", len(_vocabulary_ids(tokenizer))))
    return CompiledGuardianGrammar(
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        tokenizer_vocab_size=vocab_size,
        start_tokens=start,
        after_rationale_tokens=after_rationale,
        after_malicious_tokens=after_malicious,
        after_attacked_tokens=after_attacked,
        after_harmfulness_tokens=after_harmfulness,
        rationale_end_trie=rationale_end,
        enum_tries=enum_tries,
        safe_rationale_token_ids=safe_rationale_token_ids,
        min_rationale_content_tokens=min_rationale_content_tokens,
        max_rationale_tokens=max_rationale_tokens,
    )


class GuardianTokenFSM:
    """One response-local Guardian grammar state machine."""

    def __init__(self, grammar: CompiledGuardianGrammar) -> None:
        self.grammar = grammar
        self.state = "rationale"
        self._output_token_ids = list(grammar.start_tokens)
        self._events: list[FSMEvent] = []
        self._choice_prefix: tuple[int, ...] = ()
        self._rationale_token_count = 0
        self._rationale_content_count = 0
        self._forced_rationale_close = False
        self.judgments: dict[str, Any] = {}

    @property
    def done(self) -> bool:
        return self.state == "done"

    @property
    def events(self) -> tuple[FSMEvent, ...]:
        return tuple(self._events)

    @property
    def output_token_ids(self) -> tuple[int, ...]:
        return tuple(self._output_token_ids)

    @property
    def sampled_token_ids(self) -> tuple[int, ...]:
        return tuple(
            int(event.sampled_token_id)
            for event in self._events
            if event.sampled_token_id is not None
        )

    @property
    def forced_rationale_close(self) -> bool:
        return self._forced_rationale_close

    def output_text(self) -> str:
        try:
            return self.grammar.tokenizer.decode(
                list(self._output_token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.grammar.tokenizer.decode(list(self._output_token_ids))

    def allowed_token_ids(self) -> frozenset[int]:
        if self.state == "done":
            return frozenset()
        if self.state == "rationale":
            if self._rationale_token_count >= self.grammar.max_rationale_tokens:
                return frozenset()
            allowed = set(self.grammar.safe_rationale_token_ids)
            if self._rationale_content_count >= self.grammar.min_rationale_content_tokens:
                allowed.update(self.grammar.rationale_end_trie.allowed())
            return frozenset(allowed)
        if self.state == "rationale_end":
            return self.grammar.rationale_end_trie.allowed(self._choice_prefix)
        trie = self.grammar.enum_tries.get(self.state)
        if trie is None:
            raise RuntimeError(f"unknown FSM state: {self.state}")
        return trie.allowed(self._choice_prefix)

    def _append_fixed(self, token_ids: tuple[int, ...]) -> tuple[int, ...]:
        self._output_token_ids.extend(token_ids)
        return token_ids

    def _finish_rationale(self, *, forced: bool) -> tuple[int, ...]:
        self._choice_prefix = ()
        self.state = "malicious_user_request"
        self._forced_rationale_close = self._forced_rationale_close or forced
        return self._append_fixed(self.grammar.after_rationale_tokens)

    def force_rationale_close(self) -> FSMEvent:
        if self.state != "rationale":
            raise ConstraintViolation(f"cannot force rationale close from {self.state}")
        if self._rationale_content_count < self.grammar.min_rationale_content_tokens:
            raise ConstraintViolation("cannot force an empty/undersized rationale")
        if self._rationale_token_count < self.grammar.max_rationale_tokens:
            raise ConstraintViolation("rationale cap has not been reached")
        before = self.state
        fixed = self._finish_rationale(forced=True)
        event = FSMEvent(
            state_before=before,
            state_after=self.state,
            sampled_token_id=None,
            serialized_token_ids=(),
            deterministic_token_ids=fixed,
            semantic=False,
            forced=True,
        )
        self._events.append(event)
        return event

    def advance(self, token_id: int) -> FSMEvent:
        token_id = int(token_id)
        before = self.state
        if token_id not in self.allowed_token_ids():
            raise ConstraintViolation(
                f"{self.state}: token {token_id} is outside the allowed set"
            )

        serialized: tuple[int, ...] = ()
        deterministic: tuple[int, ...] = ()
        hidden = False

        if self.state == "rationale":
            if token_id in self.grammar.rationale_end_trie.allowed():
                self.state = "rationale_end"
                self._choice_prefix = (token_id,)
                hidden = True
                complete, _ = self.grammar.rationale_end_trie.resolved(
                    self._choice_prefix
                )
                if complete:
                    deterministic = self._finish_rationale(forced=False)
            else:
                serialized = (token_id,)
                self._output_token_ids.append(token_id)
                self._rationale_token_count += 1
                piece = _decode_piece(self.grammar.tokenizer, token_id)
                if piece.strip():
                    self._rationale_content_count += 1
        elif self.state == "rationale_end":
            self._choice_prefix += (token_id,)
            hidden = True
            complete, _ = self.grammar.rationale_end_trie.resolved(self._choice_prefix)
            if complete:
                deterministic = self._finish_rationale(forced=False)
        else:
            trie = self.grammar.enum_tries[self.state]
            self._choice_prefix += (token_id,)
            serialized = (token_id,)
            self._output_token_ids.append(token_id)
            complete, value = trie.resolved(self._choice_prefix)
            if complete:
                completed_state = self.state
                self._choice_prefix = ()
                if completed_state == "malicious_user_request":
                    self.judgments["Malicious_User_Request"] = bool(value)
                    deterministic = self._append_fixed(
                        self.grammar.after_malicious_tokens
                    )
                    self.state = "being_attacked"
                elif completed_state == "being_attacked":
                    self.judgments["Being_Attacked"] = bool(value)
                    deterministic = self._append_fixed(self.grammar.after_attacked_tokens)
                    self.state = "harmfulness_rating"
                elif completed_state == "harmfulness_rating":
                    self.judgments["Harmfulness_Rating"] = float(value)
                    deterministic = self._append_fixed(
                        self.grammar.after_harmfulness_tokens
                    )
                    self.state = "done"

        event = FSMEvent(
            state_before=before,
            state_after=self.state,
            sampled_token_id=token_id,
            serialized_token_ids=serialized,
            deterministic_token_ids=deterministic,
            semantic=True,
            hidden=hidden,
        )
        self._events.append(event)
        return event
