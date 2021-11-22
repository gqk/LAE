# -*- coding: utf-8 -*-

from ..pet.prompt import Prompt


class PromptMixin:
    prompt: Prompt

    def attach_prompt(self, prompt: Prompt):
        self.prompt = prompt

    def detach_prompt(self):
        prompt, self.prompt = getattr(self, "prompt", None), None
        return prompt

    def add_prompt(self, x):
        if isinstance(getattr(self, "prompt", None), Prompt):
            x = self.prompt(x)
        return x

    def reduce_prompt(self, x):
        if isinstance(getattr(self, "prompt", None), Prompt):
            x = self.prompt.reduce(x)
        return x
