from dataclasses import dataclass
from itertools import chain
from typing import List, Tuple, Optional, Union, cast, Any, Dict

from torch import Tensor
from torch.nn.functional import softmax
import torch
import torch.nn as nn

from common_types import Points, Vertices


class ConvexHull:
    input_size: int = 3
    special_symbol: Tensor = torch.tensor([0.0, 0.0, 1.0])


# returns final hidden state, all hidden states...
EncoderOutput = Tuple[Tensor, List[Tensor]]


@dataclass
class DecoderOutput:
    decoded_seq: List[int]
    logits: List[Tensor]
    hidden_states: List[Tensor]
    loss: Optional[List[Tensor]] = None


def get_initial_h_c(dim: int) -> Tuple[Tensor, Tensor]:
    make_tensor = lambda: torch.zeros((1, dim))
    return make_tensor(), make_tensor()


# for now, specialized to ConvexHull problem...
#
class Encoder(nn.Module):
    def __init__(self, hidden_d: int):
        super().__init__()
        self.lstm = nn.LSTMCell(ConvexHull.input_size, hidden_d, bias=True)
        self.hidden_d = hidden_d

    def forward(self, seq: Points) -> EncoderOutput:
        hidden_states: List[Tensor] = []
        # TODO:
        #   the forward call shouldn't be unsqueezing things...
        _seq = seq + [ConvexHull.special_symbol]
        h, c = self.lstm(ConvexHull.special_symbol.unsqueeze(0))
        for x in _seq:
            h, c = self.lstm(x.unsqueeze(0), (h, c))
            hidden_states.append(h)
        return h, hidden_states


class Pointer(nn.Module):
    def __init__(self, hidden_d: int, hidden_v: int):
        super().__init__()
        self.v = nn.Linear(hidden_v, 1)
        self.W_e = nn.Linear(hidden_d, hidden_v)
        self.W_d = nn.Linear(hidden_d, hidden_v)
        self.hidden_d = hidden_d
        self.hidden_v = hidden_v

    # no softmax computed here - ouput is logits of distribution...
    #
    def forward(
        self,
        decoder_state: Tensor,
        encoder_states: List[Tensor],
    ) -> Tensor:
        forward_states = [self.W_e(e) + self.W_d(decoder_state) for e in encoder_states]
        forward_states = [self.v(torch.tanh(f)) for f in forward_states]
        return torch.hstack(forward_states)


class Decoder(nn.Module):
    #
    # max_length is to prevent the forward loop from running too long...
    # "recursion limit" or something like that
    #
    max_length: int = 20

    def __init__(self, hidden_d: int, hidden_v: int):
        super().__init__()
        self.pointer = Pointer(hidden_d, hidden_v)
        self.lstm = nn.LSTMCell(ConvexHull.input_size, hidden_d)
        self.hidden_d = hidden_d
        self.hidden_v = hidden_v

    def forward(
        self,
        initial_state: Tensor,
        encoder_states: List[Tensor],
        seq: Points,  # original input sequence
        positions: Optional[List[int]] = None,
        teacher_forcing: bool = True,
    ) -> DecoderOutput:
        pointer_logits: List[Tensor] = []
        decoder_states: List[Tensor] = []
        decoded_sequence: List[int] = []
        if positions is not None:
            loss: List[Tensor] = []
        x = ConvexHull.special_symbol.unsqueeze(0)

        # _seq = [ConvexHull.special_symbol] + seq
        # stack_seq = torch.stack(_seq)
        _seq = seq + [ConvexHull.special_symbol]
        stack_seq = torch.stack(_seq)

        # this is kind of ugly / stupid
        _, c = get_initial_h_c(self.hidden_d)
        h = initial_state
        # chose_special_symbol is a flag set when the pointer distribution is used to
        # calculate the next input, and it gave highest preference to the special symbol.
        chose_special_symbol: bool = False

        #
        # there are two very different loops that occur depending on whether or not
        # positions were provided.  They have some common functionality which depend on
        # variable in this scope, so they are factored out in the next few local
        # functions.
        #

        def main_loop() -> Tensor:
            nonlocal h, c, pointer_logits, decoder_states, decoded_sequence
            h, c = self.lstm(x, (h, c))
            logits = self.pointer(h, encoder_states)
            pointer_logits.append(logits)
            decoder_states.append(h)
            dist = softmax(pointer_logits[-1], dim=1)
            index = cast(int, dist.topk(1)[1].item())
            decoded_sequence.append(index)
            if index == len(seq):
                chose_special_symbol = True
            return torch.matmul(dist, stack_seq)

        if positions is not None:
            assert loss is not None, "I don't know what happened"
            loss_fn = nn.CrossEntropyLoss()
            # must accrue loss, and stop when the sequence is over
            for position in chain(positions, [len(seq)]):
                next_x = main_loop()
                loss.append(loss_fn(pointer_logits[-1], torch.tensor([position])))
                if teacher_forcing:
                    # breakpoint()
                    x = _seq[position].unsqueeze(0)
                else:
                    x = next_x
            return DecoderOutput(decoded_sequence, pointer_logits, decoder_states, loss)
        else:
            # must use pointer-distribution to get next input, stop when
            # max_length is reached or special symbol selected
            while len(pointer_logits) < Decoder.max_length and not chose_special_symbol:
                x = main_loop()
            return DecoderOutput(decoded_sequence, pointer_logits, decoder_states)


class PointerNet(nn.Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(
        self,
        encoder: Optional[Encoder] = None,
        decoder: Optional[Decoder] = None,
        *,
        encoder_args: Optional[Dict[str, Any]] = None,
        decoder_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        if encoder is not None and encoder_args is not None:
            raise RuntimeError("only one of {encoder, encoder_args} may be provided")
        if decoder is not None and decoder_args is not None:
            raise RuntimeError("only one of {decoder, decoder_args} may be provided")
        if encoder_args is None:
            encoder_args = {}
        if encoder is None:
            encoder = Encoder(**encoder_args)
        if decoder_args is None:
            decoder_args = {}
        if decoder is None:
            decoder = Decoder(**decoder_args)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, seq: Points, **decoder_args) -> DecoderOutput:
        last_hidden, hidden_states = self.encoder(seq)
        return self.decoder(last_hidden, hidden_states, seq, **decoder_args)
