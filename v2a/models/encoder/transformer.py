import torch.nn as nn 
import torch 
import math 

class TransformerEncoder(nn.Module):

    def __init__(self,
                 input_dim,
                 query_dim,
                 heads,
                 dim_feedforward=1024,
                 n_layer=2,
                 rep_dim=None,
                 pos_encoder=None,
                 use_cls_token = True) -> None:
        super().__init__()

        self.encoder = nn.Linear(input_dim, query_dim)

        encode_layer = nn.TransformerEncoderLayer(
            d_model=query_dim,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            batch_first=True)
        self.pos_encoder = pos_encoder
        self.d_model = query_dim
        self.transformer_encoder = nn.TransformerEncoder(encode_layer,
                                                         num_layers=n_layer)
        self.decoder = nn.Linear(query_dim, rep_dim)

        self.use_cls_token = use_cls_token
        representation_token = nn.Parameter(torch.randn(1, 1, query_dim))
        self.register_parameter("representation_token", representation_token)



        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        batch, _, f = x.shape
        x = self.encoder(x)
        
        if self.use_cls_token:
            cls_token = self.representation_token.expand(batch,-1, -1)
            x = torch.cat([cls_token, x], dim = -1) ## add a representation token at sequence start
        
        
        x = x * math.sqrt(self.d_model)

        if self.pos_encoder is not None:
            x = x + self.pos_encoder(x)

        output = self.transformer_encoder(x)

        if self.use_cls_token:
            output = output[:, 0] ### only take the first token ( the representation token)
        else:
            output = output.mean(dim=1) ### mean pooling over the sequence length 
            
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the input for as many time
    steps as necessary.
    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self,
                 size: int = 0,
                 max_len: int = 100,
                 frequency=10000.0) -> None:
        """
        Positional Encoding with maximum length
        :param size: embeddings dimension size
        :param max_len: maximum sequence length
        """
        if size % 2 != 0:
            raise ValueError(
                f"Cannot use sin/cos positional encoding with odd dim (got dim={size})"
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, size, 2, dtype=torch.float) *
                              -(math.log(frequency) / size)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, size)
        super().__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        """
        Embed inputs.
        :param emb: (Tensor) Sequence of word embeddings vectors
            shape (seq_len, batch_size, dim)
        :return: positionally encoded word embeddings
        """
        # get position encodings
        return self.pe[:, :emb.size(1)]