import torch
import torch .nn as nn
import math


class StateAwarePrefixProjection(nn.Module):
    """Add task-conditioned, recency-aware state pooling to a legacy prefix.

    The historical prefix projection uses a single learned query for every
    task and prefix.  This adapter keeps that representation as its anchor but
    builds a second query from the encoded CLS token and the last valid event.
    Classification and regression own separate query offsets, residual gates,
    and recency strengths while sharing the low-rank projection weights.
    """

    TASK_INDEX = {"classification": 0, "regression": 1}

    def __init__(self, d_model, n_heads, dropout=0.1, **config):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        hidden = max(
            16,
            int(config.get("prefix_attention_hidden_dim", max(32, d_model // 2))),
        )
        self.query_projector = nn.Sequential(
            nn.Linear(d_model * 2, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=False),
            nn.LayerNorm(d_model, elementwise_affine=False),
        )
        self.task_queries = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.task_queries, std=0.02)
        self.query_norm = nn.LayerNorm(d_model)

        self.residual_projector = nn.Sequential(
            nn.Linear(d_model * 3, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(float(config.get("prefix_attention_dropout", dropout))),
            nn.Linear(hidden, d_model, bias=False),
            nn.LayerNorm(d_model, elementwise_affine=False),
        )
        self.gate_logits = nn.Parameter(
            torch.full(
                (2,), float(config.get("prefix_attention_gate_logit", -3.0))
            )
        )

        initial_recency = max(
            float(config.get("prefix_attention_initial_recency", 0.25)), 1e-4
        )
        initial_recency_logit = math.log(math.expm1(initial_recency))
        self.recency_logits = nn.Parameter(
            torch.full((2,), initial_recency_logit)
        )

    def _task_index(self, task_type):
        # Historical callers that do not specify a task retain a deterministic
        # path. All training/evaluation task encoders pass the explicit type.
        if task_type is None:
            return 0
        if task_type not in self.TASK_INDEX:
            raise ValueError(f"Unknown prefix-attention task type: {task_type}")
        return self.TASK_INDEX[task_type]

    def _lengths_and_last(self, tokens, token_mask):
        batch_size, token_count, _ = tokens.shape
        if token_mask is None:
            lengths = torch.full(
                (batch_size,), token_count, dtype=torch.long, device=tokens.device
            )
        else:
            lengths = (~token_mask).sum(dim=1).clamp_min(1)
        last_indices = (lengths - 1).clamp_max(token_count - 1)
        rows = torch.arange(batch_size, device=tokens.device)
        return lengths, tokens[rows, last_indices]

    def recency_attention_bias(
        self, token_mask, batch_size, token_count, task_type, device, dtype
    ):
        if token_mask is None:
            lengths = torch.full(
                (batch_size,), token_count, dtype=torch.long, device=device
            )
        else:
            lengths = (~token_mask).sum(dim=1).clamp_min(1)
        positions = torch.arange(token_count, device=device).unsqueeze(0)
        age = (lengths.unsqueeze(1) - 1 - positions).clamp_min(0).float()
        age = age / (lengths - 1).clamp_min(1).unsqueeze(1).float()
        task_index = self._task_index(task_type)
        strength = torch.nn.functional.softplus(self.recency_logits[task_index])
        bias = (-strength * age).to(dtype=dtype)
        if token_mask is not None:
            bias = bias.masked_fill(token_mask, float("-inf"))
        bias = bias.unsqueeze(1).expand(-1, self.n_heads, -1)
        return bias.reshape(-1, 1, token_count)

    def forward(
        self,
        cls_token,
        legacy_pooled,
        tokens,
        token_mask,
        attention,
        task_type=None,
        return_attention=False,
    ):
        _, last_event = self._lengths_and_last(tokens, token_mask)
        task_index = self._task_index(task_type)
        query = self.query_projector(torch.cat([cls_token, last_event], dim=-1))
        query = self.query_norm(query + self.task_queries[task_index]).unsqueeze(1)
        attention_bias = self.recency_attention_bias(
            token_mask,
            tokens.size(0),
            tokens.size(1),
            task_type,
            tokens.device,
            query.dtype,
        )
        state_pooled, weights = attention(
            query=query,
            key=tokens,
            value=tokens,
            attn_mask=attention_bias,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        state_pooled = state_pooled.squeeze(1)
        residual = self.residual_projector(
            torch.cat([last_event, state_pooled, legacy_pooled], dim=-1)
        )
        gate = torch.sigmoid(self.gate_logits[task_index])
        diagnostics = {
            "attention": weights,
            "gate": gate.detach(),
            "recency_strength": torch.nn.functional.softplus(
                self.recency_logits[task_index]
            ).detach(),
        }
        return gate * residual, diagnostics


class PositionalEncoding (nn .Module ):
    def __init__ (self ,d_model ,dropout =0.1 ,max_len =512 ):
        super ().__init__ ()
        self .dropout =nn .Dropout (p =dropout )
        position =torch .arange (max_len ).unsqueeze (1 )
        div_term =torch .exp (torch .arange (0 ,d_model ,2 )*(-math .log (10000.0 )/d_model ))
        pe =torch .zeros (1 ,max_len ,d_model )
        pe [0 ,:,0 ::2 ]=torch .sin (position *div_term )
        pe [0 ,:,1 ::2 ]=torch .cos (position *div_term )
        self .register_buffer ('pe',pe )
    def forward (self ,x ):
        x =x +self .pe [:,:x .size (1 ),:]
        return self .dropout (x )
class EventEncoder (nn .Module ):
    def __init__ (self ,d_model ,n_heads ,n_layers ,dropout =0.1 ,prefix_config =None ):
        super ().__init__ ()
        prefix_config =dict (prefix_config or {})
        self .pos_encoder =PositionalEncoding (d_model ,dropout )
        self .d_model =d_model
        self .cls_token =nn .Parameter (torch .zeros (1 ,1 ,d_model ))
        nn .init .normal_ (self .cls_token ,std =0.02 )
        encoder_layer =nn .TransformerEncoderLayer (
        d_model =d_model ,
        nhead =n_heads ,
        dim_feedforward =d_model *4 ,
        dropout =dropout ,
        batch_first =True ,
        activation ='gelu',
        norm_first =True
        )
        self .transformer_encoder =nn .TransformerEncoder (encoder_layer ,num_layers =n_layers )
        self .pool_query =nn .Parameter (torch .zeros (1 ,1 ,d_model ))
        nn .init .normal_ (self .pool_query ,std =0.02 )
        self .mha_pool =nn .MultiheadAttention (
        embed_dim =d_model ,
        num_heads =n_heads ,
        dropout =dropout ,
        batch_first =True
        )
        self .final_projection =nn .Linear (d_model *2 ,d_model )
        self .out_norm =nn .LayerNorm (d_model )
        self .state_aware_pool =None
        if prefix_config .get ('state_aware_prefix_attention',False ):
            self .state_aware_pool =StateAwarePrefixProjection (
            d_model ,n_heads ,dropout ,**prefix_config )
    def forward (self ,src ,src_key_padding_mask =None ,task_type =None ,return_attention =False ):
        B ,T ,D =src .shape
        cls =self .cls_token .expand (B ,1 ,D )
        src =torch .cat ([cls ,src ],dim =1 )
        if src_key_padding_mask is not None :
            pad_col =torch .zeros ((B ,1 ),dtype =torch .bool ,device =src_key_padding_mask .device )
            src_key_padding_mask =torch .cat ([pad_col ,src_key_padding_mask ],dim =1 )
        src =src *math .sqrt (self .d_model )
        src =self .pos_encoder (src )
        output =self .transformer_encoder (src ,src_key_padding_mask =src_key_padding_mask )
        cls_out =output [:,0 ,:]
        tokens =output [:,1 :,:]
        token_mask =None
        if src_key_padding_mask is not None :
            token_mask =src_key_padding_mask [:,1 :]
        pool_q =self .pool_query .expand (B ,-1 ,-1 )
        if self .state_aware_pool is None and not return_attention :
            # Preserve the historical call exactly for old configurations and
            # checkpoints. The optimized no-weight path is used by the new
            # architecture, which performs a second pooling call below.
            pooled_out ,legacy_attention =self .mha_pool (
            query =pool_q ,key =tokens ,value =tokens ,key_padding_mask =token_mask )
        else :
            pooled_out ,legacy_attention =self .mha_pool (
            query =pool_q ,
            key =tokens ,
            value =tokens ,
            key_padding_mask =token_mask ,
            need_weights =return_attention ,
            average_attn_weights =False
            )
        pooled =pooled_out .squeeze (1 )
        concatenated =torch .cat ([cls_out ,pooled ],dim =-1 )
        projected =self .final_projection (concatenated )
        encoded =self .out_norm (projected )
        state_diagnostics =None
        if self .state_aware_pool is not None :
            state_residual ,state_diagnostics =self .state_aware_pool (
            cls_out ,pooled ,tokens ,token_mask ,self .mha_pool ,
            task_type =task_type ,return_attention =return_attention )
            encoded =encoded +state_residual
        if return_attention :
            return encoded ,{
            'legacy_attention':legacy_attention ,
            'state_attention':None if state_diagnostics is None else state_diagnostics ['attention'],
            'state_gate':None if state_diagnostics is None else state_diagnostics ['gate'],
            'recency_strength':None if state_diagnostics is None else state_diagnostics ['recency_strength'],
            }
        return encoded
