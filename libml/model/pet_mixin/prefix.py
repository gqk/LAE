# -*- coding: utf-8 -*-

from ..pet.prefix import Prefix


class PrefixMixin:
    prefix: Prefix

    def attach_prefix(self, prefix: Prefix):
        self.prefix = prefix

    def detach_prefix(self):
        prefix, self.prefix = getattr(self, "prefix", None), None
        return prefix

    def add_prefix(self, key, val):
        if isinstance(getattr(self, "prefix", None), Prefix):
            key, val = self.prefix(key, val)
        return key, val

    def compensate_prefix(self, attn):
        if isinstance(getattr(self, "prefix", None), Prefix):
            attn = self.prefix.compensate(attn)
        return attn
