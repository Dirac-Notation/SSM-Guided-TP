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
        
        self.method = "full" # full / no_offloading / just_offloading / maintain_offloading / h2o
        self.token_budget = None
        
        self.select_indices = None
        self.saved_indices = None
        self.loaded_data = None
        self.attention_score = None

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
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
        
        if self.select_indices is None or self.method == "full":
            loaded_data = torch.narrow(self.data, 2, 0, self.current_length).to(tensor.device)
            loaded_data = torch.cat([loaded_data, tensor], dim=-2)
            
        else:
            if self.method == "streamingllm" or self.method == "ssm-guided":
                loaded_data = self.data.index_select(dim, self.select_indices[:-59])
                loaded_data = torch.cat([loaded_data, tensor], dim=-2)
            
            elif self.method == "just_offloading":
                loaded_data = self.data.index_select(dim, self.select_indices[:-59].to("cpu")).to(tensor.device)
                loaded_data = torch.cat([loaded_data, tensor], dim=-2)
            
            elif self.method == "maintain_offloading":
                select_indices = self.select_indices[:-59]
                
                if self.saved_indices is None:
                    loaded_data = self.data.index_select(dim, select_indices.to("cpu")).to(tensor.device)
                    self.loaded_data = torch.cat([loaded_data, tensor], dim=-2)
                
                    self.saved_indices = self.select_indices
                else:
                    saved_indices = self.saved_indices[:-59]
                    alredy_loaded_tokens = custom_isin1d(saved_indices, select_indices)
                    need_load_tokens = ~custom_isin1d(select_indices, saved_indices)

                    alredy_loaded_indices = saved_indices[alredy_loaded_tokens]
                    need_load_indices = select_indices[need_load_tokens]
                    
                    loaded_data = self.data.index_select(dim, need_load_indices.to("cpu")).to(tensor.device)
                    
                    self.loaded_data = torch.cat([
                        self.loaded_data[:,:,:-59,:][:,:,alredy_loaded_tokens,:],
                        loaded_data,
                        tensor
                    ], dim=dim)
                    
                    self.saved_indices = torch.cat([
                        alredy_loaded_indices,
                        need_load_indices,
                        self.select_indices[-59:]
                    ])

            elif self.method == "h2o":
                loaded_data = self.data.gather(dim, self.select_indices[:,:,:-59].unsqueeze(-1).expand(-1,-1,-1,self.data.size(-1)))
                loaded_data = torch.cat([loaded_data, tensor], dim=-2)
                
        dst = self.data.narrow(dim, self.current_length, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        
        return self.loaded_data if self.loaded_data is not None else loaded_data
    
    def h2o_after_verifying(self, verified_indices: torch.Tensor = None):
        if self.method != "h2o":
            return
        
        recent_budget = int(self.token_budget/2)
        hh_budget = recent_budget
        
        if verified_indices is None:
            self.acculmulated_score = self.attention_score.sum(dim=-2)
            select_indices = self.acculmulated_score[:,:,:-recent_budget].topk(hh_budget).indices.sort().values
            self.select_indices = torch.cat([select_indices, torch.arange(self.current_length-recent_budget, self.current_length+59, device=select_indices.device).view(1,1,-1).expand(-1,32,-1)], dim=-1)
        else:
            num_new_tokens = verified_indices.numel()
            token_len = self.current_length - 59 + num_new_tokens
            verified_idx = verified_indices - token_len + 1
            
            # 스코어 최신화
            acculmulated_score = self.attention_score[:,:,verified_idx,:].sum(dim=-2)
            new_score = torch.zeros(
                (self.acculmulated_score.size(0), self.acculmulated_score.size(1), token_len),
                device=self.acculmulated_score.device,
                dtype=self.acculmulated_score.dtype
            ).scatter(-1, self.select_indices[:,:,:-59+num_new_tokens], acculmulated_score)
            new_score[:,:,:self.acculmulated_score.size(-1)] += self.acculmulated_score
            self.acculmulated_score = new_score
            
            # 인덱스 최신화화
            select_indices = self.acculmulated_score[:,:,:-recent_budget].topk(hh_budget).indices.sort().values
            self.select_indices = torch.cat([select_indices, torch.arange(token_len-recent_budget, token_len+59, device=select_indices.device).view(1,1,-1).expand(-1,32,-1)], dim=-1)
            

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
