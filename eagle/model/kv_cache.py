import torch
import math

from .utils import Timer

def custom_isin1d(test_tensor: torch.Tensor, target_tensor: torch.Tensor):
    if test_tensor.dim() != 1 or target_tensor.dim() != 1:
        assert "custom_isin1d must be used for 1d tensors"
    return (test_tensor.unsqueeze(1) == target_tensor).sum(dim=1).bool()

class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.
        current_length (int): Current length of the data being stored.
    """

    def __init__(self, data, current_length):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int): Initial length of the data.
        """
        self.data = data
        self.current_length = current_length
        
        self.method = None # full / no_offloading / just_offloading / maintain_offloading / h2o
        self.token_budget = None
        self.revive_budget = None
        
        self.prefill = False
        
        self.select_indices = None
        self.saved_indices = None
        self.loaded_data = None
        self.acculmulated_score = None
        
        self.ssm_indices = None
        self.forgetting_factor = 1.0

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length,
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.

        Args:
            indices (torch.Tensor): Indices of the data tensor to be copied.
            prev_length (int): Previous length before adding new data.
            dim (int, optional): Dimension along which copying should be performed. Default is 2.
        """
        tgt = self.data.index_select(dim, indices)
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation should be done. Default is 2.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
            
        if not self.prefill:
            loaded_data = self.data.narrow(dim, 0, self.current_length).to(tensor.device)
            if self.method != "full":
                self.prefill = True
        elif self.method == "streamingllm":
            loaded_data = torch.cat((
                self.data.narrow(dim, 0, 4),
                self.data.narrow(dim, self.current_length-(self.token_budget-4), self.token_budget-4)
            ), dim=dim)
            self.select_indices = 0
        elif self.method == "streamingssm":
            loaded_data = torch.cat((
                self.data.narrow(dim, 0, 4),
                self.data.index_select(dim, self.ssm_indices),
                self.data.narrow(dim, self.current_length-(self.token_budget-4), self.token_budget-4)
            ), dim=dim)
            self.select_indices = 0
        elif self.method == "h2o":
            loaded_data = self.data.gather(dim, self.select_indices.unsqueeze(-1).expand(-1, -1, -1, self.data.size(-1)))
        elif self.method == "h2ossm":
            if self.ssm_indices is not None:
                self.select_indices = torch.cat((
                    self.ssm_indices.view(1,1,-1).repeat(*self.select_indices.shape[:2],1),
                    self.select_indices
                ), dim=2)
                self.acculmulated_score = torch.cat((
                    torch.zeros((*self.select_indices.shape[:2], *self.ssm_indices.shape), dtype=self.acculmulated_score.dtype, device=self.acculmulated_score.device),
                    self.acculmulated_score
                ), dim=2)
            
            loaded_data = self.data.gather(dim, self.select_indices.unsqueeze(-1).expand(-1, -1, -1, self.data.size(-1)))

        dst = self.data.narrow(dim, self.current_length, tensor.size(dim))
        dst.copy_(tensor)
        self.current_length.add_(tensor.size(dim))
        
        return self.loaded_data if self.loaded_data is not None else torch.cat([loaded_data, tensor], dim=dim)
    
    def h2o_update(self, attention_score: torch.Tensor):
        if self.method != "h2o":
            return

        if self.select_indices is None:
            recent_budget = self.token_budget // 2
            hh_budget = recent_budget
            acc_score = ((self.forgetting_factor**torch.arange(0, attention_score.size(2), dtype=attention_score.dtype, device=attention_score.device).flip(dims=[0]).view(1,1,-1,1))*attention_score).sum(dim=-2)
            select_idx = acc_score[:, :, :-recent_budget].topk(hh_budget).indices.sort().values
            extra_idx = torch.arange(
                self.current_length - recent_budget,
                self.current_length,
                device=select_idx.device
            ).view(1, 1, -1).expand_as(select_idx)
            self.select_indices = torch.cat((select_idx, extra_idx), dim=-1)
            self.acculmulated_score = acc_score.gather(dim=2, index=self.select_indices)
        else:
            self.attention_score = attention_score

    def h2o_after_verifying(self, select_indices: torch.Tensor):
        if self.method != "h2o":
            return

        num_new = select_indices.numel()
        recent_budget = self.token_budget // 2
        zero_idx = select_indices - select_indices.min()
        all_part = self.attention_score[:, :, zero_idx, :]
        cache_part = all_part[:, :, :, :self.acculmulated_score.size(2)]
        select_part = all_part[:, :, :, self.acculmulated_score.size(2):][:, :, :, zero_idx]
        self.attention_score = torch.cat((cache_part, select_part), dim=3)
        acc_score = self.attention_score.sum(dim=2)
        acc_score[:, :, :self.acculmulated_score.size(2)] += self.forgetting_factor*self.acculmulated_score
        self.acculmulated_score = acc_score
        
        new_select = self.acculmulated_score[:, :, :-recent_budget].topk(recent_budget, dim=2).indices.sort().values
        extra_idx = torch.arange(
            self.current_length - 59 - recent_budget + num_new,
            self.current_length - 59 + num_new,
            device=self.select_indices.device
        ).view(1, 1, -1).expand_as(new_select)
        self.select_indices = torch.cat((self.select_indices.gather(dim=2, index=new_select), extra_idx), dim=2)
        self.acculmulated_score = torch.cat((
            self.acculmulated_score.gather(dim=2, index=new_select),
            self.acculmulated_score[:, :, -recent_budget:]
        ), dim=2)
            

def initialize_past_key_values(model):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to store and reuse
    past key and value states during autoregressive decoding, which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states need to be initialized.

    Returns:
        tuple:
            - past_key_values (list): A list of KVCache objects for each layer in the model.
            - past_key_values_data (torch.Tensor): The tensor that will store all keys and values.
            - current_length_data (torch.Tensor): A tensor tracking the current length of keys/values in the cache.
    """
    # Extracting configuration from the model
    config = model.config
    # Initializing the batch size to 1, this can be modified if different batch sizes are required
    batch_size = 1
    # Initializing a tensor to store past keys and values for all layers

    devices=[]
    for i in range(config.num_hidden_layers):
        try:
            device = model.model.layers[i].self_attn.q_proj.weight.device
        except:
            device = model.layers[i].self_attn.q_proj.weight.device
        devices.append(device)
        # devices.append(torch.device("cpu"))
    past_key_values_data_list=[]
    startnum=0
    startdevice=devices[0]
    for id,i in enumerate(devices):
        if startdevice!=i:
            past_key_values_data = torch.zeros(
                startnum * 2,
                batch_size,
                config.num_key_value_heads,
                config.max_position_embeddings,
                config.hidden_size // config.num_attention_heads,
                device=startdevice,
                dtype=model.dtype,
            )
            past_key_values_data_list.append(past_key_values_data)
            startdevice = i
            startnum=0
        startnum += 1
    past_key_values_data = torch.zeros(
        startnum * 2,
        batch_size,
        config.num_key_value_heads,
        config.max_position_embeddings,
        config.hidden_size // config.num_attention_heads,
        device=startdevice,
        dtype=model.dtype,
    )
    past_key_values_data_list.append(past_key_values_data)
    # Initialize tensor to store the current length of the cached data for all layers.
    # [IMPORTANT] It needs to be kept on CPU for quick access and updates.
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    # Creating a KVCache for each pair of key and value in all layers
    past_key_values = [] * config.num_hidden_layers

    bias=0
    start_data_m=devices[0].index
    for i in range(config.num_hidden_layers):
        data_m=devices[i].index
        if data_m!=start_data_m:
            bias=0
            start_data_m=data_m
        try:
            past_key_values.append(
                [
                    KVCache(past_key_values_data_list[data_m-devices[0].index][2*bias + j], current_length_data[i * 2 + j])
                    for j in range(2)
                ]
            )
        except:
            past_key_values.append(
                [
                    KVCache(past_key_values_data_list[0][2 * bias + j],
                            current_length_data[i * 2 + j])
                    for j in range(2)
                ]
            )
        bias+=1
    return past_key_values, past_key_values_data_list, current_length_data
