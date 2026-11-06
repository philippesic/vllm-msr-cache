"""Fixed-token forcing for explicitly labelled diagnostic replay, never live runs."""
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates


class FixedTokenReplay(LogitsProcessor):
    @classmethod
    def validate_params(cls, params):
        tokens = (params.extra_args or {}).get('diagnostic_replay_tokens')
        if tokens is not None and (not isinstance(tokens, list) or not tokens
                or any(type(t) is not int or t < 0 for t in tokens)):
            raise ValueError('diagnostic_replay_tokens must be a nonempty list of token IDs')

    def __init__(self, vllm_config, device, is_pin_memory):
        self.requests = {}

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        def state(params, prompt, output):
            tokens = (params.extra_args or {}).get('diagnostic_replay_tokens')
            return (tokens, output) if tokens is not None else None
        process_dict_updates(self.requests, batch_update, state)

    def apply(self, logits):
        for index, (tokens, output) in self.requests.items():
            position = len(output)
            if position >= len(tokens):
                raise RuntimeError('Replay generated beyond frozen continuation')
            token = tokens[position]
            if token >= logits.shape[-1]:
                raise ValueError('Replay token outside vocabulary')
            logits[index].fill_(-float('inf'))
            logits[index, token] = 0
        return logits
