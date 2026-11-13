import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lm.utils import count_params

"""
Dimension symbols:
    B - batch size
    S - sequence length
    D - hidden dimension (n_embd)
    H - number of attention heads (n_head)
    HD - hidden dimension of a single attention head (d // n_head)
    V - size of the vocabulary
"""


class MultiHeadAttention(nn.Module):
    """The multi-head attention module in a decoder block."""

    def __init__(self, n_embd: int, n_head: int, p_dropout: float = 0.1):
        super().__init__()
        """Initialize the modules used by multi-head attention."""

        self.n_head = n_head
        attn_hidden_dim = n_embd // n_head

        self.q_attn = nn.Linear(n_embd, n_embd)
        self.k_attn = nn.Linear(n_embd, n_embd)
        self.v_attn = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(p_dropout)

        scale_factor = 1 / torch.sqrt(torch.tensor(attn_hidden_dim))
        self.register_buffer("scale_factor", scale_factor)
    
    def q_kT_v(self, x: torch.FloatTensor
        ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    
        q = rearrange(self.q_attn(x), "b s (h hd) -> b h s hd", h=self.n_head)
        kT = rearrange(self.k_attn(x), "b s (h hd) -> b h hd s", h=self.n_head)
        v = rearrange(self.v_attn(x), "b s (h hd) -> b h s hd", h=self.n_head)

        return q, kT, v

    def self_attention(
        self,
        q: torch.FloatTensor,
        kT: torch.FloatTensor,
        v: torch.FloatTensor,
        attention_mask: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        # 1. Scaled dot product (only multiply by self.scale_factor)
        unmasked_attn_logits = torch.matmul(q, kT) * self.scale_factor

        # 2. Causal mask: (S, S) -> (1, 1, S, S)
        S = q.size(-2)
        causal_mask = torch.tril(
            torch.ones(S, S, dtype=torch.bool, device=q.device)
        )
        mask = causal_mask.unsqueeze(0).unsqueeze(1)

        # 3. Combine with attention_mask: (B, S) -> (B, 1, S, S)
        if attention_mask is not None:
            attn_mask_bool = attention_mask.to(device=q.device, dtype=torch.bool)
            # Outer product ensures both padded queries and padded keys are False
            pad_mask_2d = attn_mask_bool.unsqueeze(2) & attn_mask_bool.unsqueeze(1)
            mask = mask & pad_mask_2d.unsqueeze(1)

        # 4. Fill False positions with float_min (prevents NaNs on all-False rows)
        float_min = torch.finfo(q.dtype).min
        attn_logits = unmasked_attn_logits.masked_fill(~mask, float_min)

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. Multiply by v and merge heads: (B, H, S, HD) -> (B, S, D)
        attn = torch.matmul(attn_weights, v)
        attn = rearrange(attn, "b h s hd -> b s (h hd)")

        return attn

    def projection(self, attn: torch.FloatTensor) -> torch.FloatTensor:
        """Apply a dropout and a linear projection to outputs of attention"""
        return self.dropout(self.proj(attn))

    def forward(
        self, x: torch.FloatTensor, attention_mask: torch.FloatTensor | None = None
    ) -> torch.FloatTensor:
        """A full forward pass of the multi-head attention module."""
        q, kT, v = self.q_kT_v(x)
        attn = self.self_attention(q, kT, v, attention_mask=attention_mask)
        y = self.projection(attn)
        return y


class FeedForward(nn.Module):
    """The feedforward attention module in a decoder block."""

    def __init__(self, n_embd: int, p_dropout: float = 0.1):
        """Initialize the modules used by feedforward."""
        super().__init__()

        middle_dim = 4 * n_embd  # stick to what GPT-2 does
        self.linear_in = nn.Linear(n_embd, middle_dim)
        self.linear_out = nn.Linear(middle_dim, n_embd)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """A full forward pass of the feedforward module."""
        y = F.gelu(self.linear_in(x))
        z = self.dropout(self.linear_out(y))
        return z


class DecoderBlock(nn.Module):
    """A single decoder block in a decoder language model."""

    def __init__(self, n_embd: int, n_head: int):
        """Initialize the modules used in a decoder block."""
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd)
        self.mha = MultiHeadAttention(n_embd, n_head)
        self.ff = FeedForward(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

    def forward(
        self, x: torch.FloatTensor, attention_mask: torch.FloatTensor | None
    ) -> torch.FloatTensor:
        """A full forward pass of the decoder block.

        Args:
            x: embeddings or hidden states (B x S x D) from the previous decoder block
            attention_mask (optional): Mask indicating tokens that shouldn't
              be included in self-attention (B x S). 1 stands for a token that is
              included, and 0 stands for a token that isn't.
        Returns:
            y: outputs of the current decoder block

        Different from what you saw in class which uses ReLU as the activation,
        we are going to follow GPT-2 which uses GeLU. You should also apply
        self.dropout to the output.

        A note on where to place layer normalization (LN): in the lecture, you
        saw "post-LN", which applies LN to the outputs of MHA / FF modules after
        the residual is added. Another approach to do this is "pre-LN", which
        appiles LN to the inputs of the attention and feedforward modules. Both
        implementations should pass the tests. See explanations here:
        https://sh-tsang.medium.com/review-pre-ln-transformer-on-layer-normalization-in-the-transformer-architecture-b6c91a89e9ab
        """
        """A full forward pass of the decoder block."""
        # 1. Pre-LN + Multi-Head Attention + Residual Connection
        x = x + self.mha(self.ln_1(x), attention_mask=attention_mask)
        
        # 2. Pre-LN + FeedForward + Residual Connection
        y = x + self.ff(self.ln_2(x))
        
        return y


class DecoderLM(nn.Module):
    """The decoder language model."""

    def __init__(
        self,
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        n_layer: int,
        p_dropout: float = 0.1,
    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_positions = n_positions
        self.n_layer = n_layer
        self.p_dropout = p_dropout

        self.token_embeddings = nn.Embedding(n_vocab, n_embd)
        self.position_embeddings = nn.Embedding(n_positions, n_embd)
        self.blocks = nn.ModuleList(
            [DecoderBlock(n_embd, n_head) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(self.p_dropout)

        # initialize weights according to nanoGPT
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / torch.sqrt(2 * n_layer))

        # count flops per token according to nanoGPT
        self.flops_per_token = (
            6 * count_params(self) + 12 * n_layer * n_embd * n_positions
        )

    def embed(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """Convert input_ids to embeddings (token_embeddings + positional_embeddings).

        Args:
            input_ids: tokens ids with shape (B x S)
            attention_mask (optional): Mask indicating whether tokens should be
              ignored.

        Returns:
            embeddings: token representations with shape (B x S x D)
        """

        """
        Position ids are indices of tokens in the sequence. When attention_mask
        isn't provided, they are simply [0, 1, 2, ...] for every sequence in the
        batch. When they are provided, you should ignore tokens with attention_mask
        equal to 0.
        
        Example (B = 2, S = 5):
        
        attention_mask = tensor([
         [0., 0., 1., 1., 1.],
         [1., 1., 1., 1., 1.]
        ])

        position_ids = tensor([,
         [0, 1, 2, 3, 4]
        ])

        Note that the position ids for masked out tokens do not matter, as long
        as they don't trigger out-of-bounds errors when fed into the embedding
        layer. I.e., they should be within [0, n_positions).

        Hint: torch.cumsum
        """

        assert input_ids.shape[1] <= self.n_positions
        B, S = input_ids.shape
        token_embeddings = self.token_embeddings(input_ids)

        if attention_mask is None:
            position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        else:
            position_ids = (
                (torch.cumsum(attention_mask, dim=-1) - 1)
                .clamp(min=0)
                .long()
                .to(input_ids.device)
            )

        positional_embeddings = self.position_embeddings(position_ids)
        return self.dropout(token_embeddings + positional_embeddings)

    def token_logits(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Project the final hidden states of the model to token logits.

        Args:
            x: hidden states produced by the final decoder block (B x S x D)

        Returns:
            logits: logits corresponding to the predicted next token likelihoods (B x S x V)

        Hint: Use the weight tying technique discussed in Q1.2
        """

        logits = F.linear(x, self.token_embeddings.weight)
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """A forward pass of the decoder LM, converting input_ids to token logits.

        Args:
            input_ids: tokens ids with shape (B x S)
            attention_mask (optional): Mask indicating whether tokens should be
              ignored.

        Returns:
            logits: logits corresponding to the predicted next token likelihoods (B x S x V)
        """

        x = self.embed(input_ids, attention_mask=attention_mask)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.ln(x)
        logits = self.token_logits(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
