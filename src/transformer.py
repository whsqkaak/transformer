import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import copy
from typing import Optional

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForwardNet


class Transformer(nn.Module):
    """
    This class is implementation of transformer architecture
    proposed in "Attention Is All You Need"(Vaswani et al., 2017).
    
    Args:
        dim_model:
            The dimension of model, the number of expected features in the encoder/decoder inputs.
        num_tokens:
            The number of tokens, the number of classes, the number of embeddings.
        num_heads:
            The number of attention heads in the multi head attention models.
        num_encoder_layers:
            The number of sub-encoder layers in the encoder.
        num_decoder_layers:
            The number of sub-decoder layers in the decoder.
        dim_ffn:
            The dimension of the feed forward network model.
        dropout:
            The dropout value.
    """
    
    def __init__(
        self,
        dim_model: int = 512,
        num_tokens: int,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_ffn: int = 2048,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        # Transformer Encoder part
        self.encoder_embedding = Embedding(num_tokens, dim_model)
        self.encoder_pos = PositionalEncoding(dim_model, dropout)
        encoder_layer = TransformerEncoderLayer(dim_model, num_heads, dim_ffn, dropout)
        encoder_norm = nn.LayerNorm(dim_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        # Transformer Decoder part
        self.decoder_embedding = Embedding(num_tokens, dim_model)
        self.decoder_pos = PositionalEncoding(dim_model, dropout)
        decoder_layer = TransformerDecoderLayer(dim_model, num_heads, dim_ffn, dropout)
        decoder_norm = nn.LayerNorm(dim_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        self.output_layer = nn.Linear(dim_model, num_tokens)
     
    def forward(
        self,
        source: Tensor,
        target: Tensor,
        attn_mask_source: Optional[Tensor] = None,
        attn_mask_target: Optional[Tensor] = None,
        attn_mask_encoder_output: Optional[Tensor] = None
    ) -> Tensor:
        """
        Take in and process masked source/target sequences.
        
        Args:
            source:
                The Sequence to the encoder.
            target:
                The Sequence to the decoder.
            attn_mask_source:
                The attention mask for the source seqeunce.
            attn_mask_target:
                The attention mask for the target sequence.
            attn_mask_encoder_output:
                The attention mask for the encoder output sequence.
                
        Shape:
            source: `(B, S, E)`
            target: `(B, T, E)`
            attn_mask_source: `(S, S)` or `(B, S, S)`
            attn_mask_target: `(T, T)` or `(B, T, T)`
            attn_mask_encoder_output: `(T, S)` or `(B, T, S)`
            
            output: `(B, T, E)`
                
            where 
                B is batch size.
                S is a length of the source sequence.
                E is the embedding dimension.
                T is a length of the target sequence.
                
        """
        # Transformer Encoder part
        source = self.encoder_embedding(source)
        source = self.encoder_pos(source)
        encoder_output = self.encoder(source, attn_mask=attn_mask_source)
        
        # Transformer Decoder part
        target = self.decoder_embedding(target)
        target = self.decoder_pos(target)
        output = self.decoder(target,
                              encoder_output,
                              attn_mask_inputs=attn_mask_target,
                              attn_mask_encoder_output=attn_mask_encoder_output)
        
        # Transformer Output part
        output = self.output_layer(output)
        
        return F.log_softmax(output, dim=-1)
        
    
class TransformerEncoder(nn.Module):
    """
    The TransformerEncoder is composed of a stack of N identical encoder layers.
    
    Args:
        encoder_layer:
            An instance of the TransformerEncoderLayer() class.
        num_layers:
            The number of sub-encoder layers in the encoder.
        norm:
            The layer normalization component.
    """
    
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int = 6,
        norm: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.norm = norm
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
    
    def forward(
        self,
        inputs: Tensor,
        attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Pass the input through the encoder layers.
        
        Args:
            inputs:
                The input sequence to the encoder.
            attn_mask:
                The mask for the input sequence.
        """
        output = inputs
        for layer in self.layers:
            output = layer(output, attn_mask)
            
        if self.norm is not None:
            output = self.norm(output)
            
        return output
    

class TransformerEncoderLayer(nn.Module):
    """
    The TransformerEncoderLayer has two sub-layers.
    The first is a multi-head self-attention mechanism.
    The second is a simple, postion-wise fully connected feed-forward network.
    
    Args:
        dim_model:
            The dimension of model, the number of expected features in the encoder/decoder inputs.
        num_heads:
            The number of attention heads in the multi head attention models.
        dim_ffn:
            The dimension of the feed forward network model.
        dropout:
            The dropout value.
    """
    
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_ffn : int = 2048,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim_model) # Normalization with multi head self attn
        self.norm2 = nn.LayerNorm(dim_model) # Normalization with ffn
        self.multi_head_self_attn = MultiHeadAttention(dim_model, num_heads)
        self.ffn = PositionwiseFeedForwardNet(dim_model, dim_ffn, dropout)
        
    def forward(
        self,
        inputs: Tensor,
        attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Pass the input through the encoder layer.
        
        Args:
            inputs:
                The Sequence to the encoder layer.
            attn_mask:
                The mask for the input sequence.
                
        Shape:
            inputs: `(B, N, E)`
            attn_mask: either a 3D tensor of shape `(B, N, N)` or
                a 2D tensor of shape `(N, N)`
            
            Returns: `(B, N, E)`
            
            where
                B is batch size,
                N is a length of the input sequence,
                E is embedding dimension.
                
        """
        residual = inputs
        attn_value = self.multi_head_self_attn(inputs, inputs, inputs, attn_mask)[0]
        x = self.norm1(residual + attn_value)
        
        residual = x
        ffn_output = self.ffn(x)
        output = self.norm2(residual + ffn_output)
        
        return output
    
    
class TransformerDecoder(nn.Module):
    """
    The TransformerDecoder is also composed of a stack of N identical layers.
    
    Args:
        decoder_layer:
            An instance of the TransformerDecoderLayer() class.
        num_layers:
            The number of sub-decoder layers in the decoder.
        norm:
            The layer normalization component.
    """
    
    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int = 6,
        norm: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.norm = norm
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        
    def forward(
        self,
        inputs: Tensor,
        encoder_output: Tensor,
        attn_mask_inputs: Optional[Tensor] = None,
        attn_mask_encoder_output: Optional[Tensor] = None
    ) -> Tensor:
        """
        Pass the inputs through the decoder.
        
        Args:
            inputs:
                The sequence to the decoder.
            encoder_outputs:
                The sequence from the last layer of the encoder.
            attn_mask_inputs:
                The attention mask for the inputs sequence.
            attn_mask_encoder_output:
                The attention mask for the encoder outputs sequence.
        """
        output = inputs
        
        for layer in self.layers:
            output = layer(output, 
                           encoder_output,
                           attn_mask_inputs=attn_mask_inputs,
                           attn_mask_encoder_output=attn_mask_encoder_output)
        
        if self.norm is not None:
            output = self.norm(output)
            
        return output
    
    
class TransformerDecoderLayer(nn.Module):
    """
    The TransformerDecoderLayer has three sub-layers.
    The first is a multi-head self-attention mechanism.
    The second is a multi-head attenion mechanism over the output of the encoder.
    The third is a simple, postion-wise fully connected feed-forward network.
    
    Args:
        dim_model:
            The dimension of model, the number of expected features in the encoder/decoder inputs.
        num_heads:
            The number of attention heads in the multi head attention models.
        dim_ffn:
            The dimension of the feed forward network model.
        dropout:
            The dropout value.
    """
    
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_ffn : int = 2048,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim_model) # Normalization with multi head self attn
        self.norm2 = nn.LayerNorm(dim_model) # Normalization with multi head attn
        self.norm3 = nn.LayerNorm(dim_model) # Normalization with ffn
        self.multi_head_self_attn = MultiHeadAttention(dim_model, num_heads)
        self.multi_head_attn = MultiHeadAttention(dim_model, num_heads)
        self.ffn = PositionwiseFeedForwardNet(dim_model, dim_ffn, dropout)
        
    def forward(
        self,
        inputs: Tensor,
        encoder_output: Tensor,
        attn_mask_inputs: Optional[Tensor] = None,
        attn_mask_encoder_output: Optional[Tensor] = None
    ) -> Tensor:
        """
        Pass the inputs through the decoder layer.
        
        Args:
            inputs:
                The sequence to the decoder layer.
            encoder_outputs:
                The sequence from the last layer of the encoder.
            attn_mask_inputs:
                The attention mask for the inputs sequence.
            attn_mask_encoder_output:
                The attention mask for the encoder outputs sequence.
        """
        # Self-attention block
        residual = inputs
        self_attn_value = self.multi_head_self_attn(inputs, inputs, inputs, attn_mask_inputs)[0]
        x = self.norm1(residual + self_attn_value)
        
        # Multihead attention block
        residual = x
        attn_value = self.multi_head_attn(x, encoder_outputs, encoder_outputs, attn_mask_encoder_output)[0]
        x = self.norm2(residual + attn_value)
        
        # Feed Forwawrd block
        residual = x
        ffn_output = self.ffn(x)
        x = self.norm3(residual + ffn_output)
        
        return output
        

class Embedding(nn.Module):
    """
    Similarly to other sequence transduction models,
    We use learned embeddings to convert the input tokens and output tokens to vectors of dimension.
    We also use the usual learned linear transformation and softmax function to convert
    the decoder output to predicted next-token probabilityies.
    In the embedding layers, transformer multiply those weights by sqrt(dim_model).
    
    Args:
        num_embeddings:
            The number of embeddings.
        dim_model:
            The dimension of model, the number of expected features in the encoder/decoder inputs.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        dim_model: int
    ) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.embedding = nn.Embedding(num_embedding, dim_model)
        
    def forward(
        self,
        inputs: Tensor
    ) -> Tensor:
        return self.embedding(inputs) * math.sqrt(self.dim_model)
    
    
class PositionalEncoding(nn.Module):
    """
    Since our model contains no recurrence and no convolution,
    in order for the model to make use of the order of the sequence,
    we must inject some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension(dim_model) as the embeddings,
    so that the two can be summed.
    
    In this work, we use sine and cosine functions of different frequencies:
    
        PositionalEncoding(pos, 2i) = sin(pos / 10000^{2i/dim_model})
        PositionalEncoding(pos, 2i+1) = cos(pos / 10000^{2i/dim_model})
    
    where pos is the position and i is the dimension.
    
    Note::
        
        1 / (10000^(2i/dim_model)) = exp(-log(10000^(2i/dim_model)))
                                   = exp(-2i / dim_model * log(10000))
                                   = exp(2i * -(log(10000) / dim_model))
                                   
    Args:
        dim_model:
            The dimension of model, the number of expected features in the encoder/decoder inputs.
        dropout:
            The dropout value.
        max_len:
            The max length of the incoming sequence.
    """
    
    def __init__(
        self,
        dim_model: int,
        dropout: float = 0.1,
        max_len: int = 5000
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # First, initialize positional encoding to zeros.
        pe = torch.zeros(max_len, dim_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # torch.arange(0, dim_model, 2) == 2i
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        
        # pe(pos, 2i) and pe(pos, 2i+1)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Now pe has shape of (1, max_len, dim_model).
        pe = pe.unsqueeze(0)
        self.pe = pe
        
    def forward(
        self,
        inputs: Tensor
    ) -> Tensor:
        """
        Add Positional Encoding.
        
        Args:
            inputs:
                The input sequence.
        
        Shape:
            inputs: `(B, N, E)` where
                B is batch size.
                N is a length of input sequence.
                E is a embedding dimension.
        """
        outputs = inputs + self.pe[:inputs.shape[1], :]
        return self.dropout(outputs)
        