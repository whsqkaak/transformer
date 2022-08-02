from torch import nn

class PositionwiseFeedForwardNet(nn.Module):
    """
    This class is a Position-wise fully connected Feed-Forward Network
    which is apllied to each position separately and identically.
    This consists of two linear transformations with a ReLU activation in between.
    
    Args:
        dim_model:
            The dimension of model, the number of expected features in the encoder/decoder inputs.
        dim_ffn:
            The dimension of the feed forward network model.
        dropout:
            The dropout value.
    """
    
    def __init__(
        self,
        dim_model: int = 512,
        dim_ffn : int = 2048,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_ffn), # x * W_1 + b_1
            nn.ReLU(), # max((x * W_1 + b_1), 0)
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, dim_model), # max((x * W_1 + b_1), 0) * W_2 + b_2
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        inputs: Tensor
    ) -> Tensor:
        return self.ffn(inputs)