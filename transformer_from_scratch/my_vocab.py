from collections import Counter, OrderedDict

class Vocab:
    def __init__(self, tokens):
        self.itos = list(tokens)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.default_index = None

    def __getitem__(self, token):
        if token in self.stoi:
            return self.stoi[token]
        elif self.default_index is not None:
            return self.default_index
        else:
            raise RuntimeError(f"Token '{token}' not in vocabulary and default index not set.")

    def __len__(self):
        return len(self.itos)

    def set_default_index(self, index):
        self.default_index = index

    def get_itos(self):
        return self.itos

    def get_stoi(self):
        return self.stoi


def vocab(ordered_dict, min_freq=1, specials=None, special_first=True):
    specials = specials or []

    for token in specials:
        ordered_dict.pop(token, None)

    tokens = []
    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            tokens.append(token)

    if special_first:
        tokens = list(specials) + tokens
    else:
        tokens = tokens + list(specials)

    return Vocab(tokens)


def build_vocab_from_iterator(
    iterator,
    min_freq=1,
    specials=None,
    special_first=True,
    max_tokens=None
):
    counter = Counter()

    for tokens in iterator:
        counter.update(tokens)

    specials = specials or []

    sorted_by_freq = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    if max_tokens is not None:
        assert len(specials) < max_tokens, "special tokens >= max_tokens"
        sorted_by_freq = sorted_by_freq[: max_tokens - len(specials)]

    ordered_dict = OrderedDict(sorted_by_freq)

    return vocab(
        ordered_dict,
        min_freq=min_freq,
        specials=specials,
        special_first=special_first
    )
