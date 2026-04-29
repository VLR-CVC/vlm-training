import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

class TokenDispatcher:
    """Consolidated EP/SP dispatcher. Handles local token reorder and all-to-all."""
    
    def __init__(self, num_experts: int, top_k: int, score_before_experts: bool = True):
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_before_experts = score_before_experts
        
        self.ep_mesh: DeviceMesh | None = None
        self.sp_size: int = 1
        self.sp_rank: int = -1

    def _split_along_sp(self, *tensors: torch.Tensor) -> list[torch.Tensor]:
        results = []
        for t in tensors:
            local_num_tokens = t.shape[0] // self.sp_size
            offset = self.sp_rank * local_num_tokens
            results.append(t[offset : offset + local_num_tokens])
        return results

    def _permute(self, routed_input, num_tokens_per_expert_group, ep_size, num_local_experts):
        device = num_tokens_per_expert_group.device
        total = num_tokens_per_expert_group.sum().item()

        t_mat = num_tokens_per_expert_group.view(ep_size, num_local_experts)
        input_starts = (num_tokens_per_expert_group.cumsum(0) - num_tokens_per_expert_group).view(ep_size, num_local_experts)

        segment_lens = t_mat.t().reshape(-1)
        input_starts = input_starts.t().reshape(-1)

        seg_ids = torch.arange(segment_lens.shape[0], device=device).repeat_interleave(segment_lens.long())
        output_starts = segment_lens.cumsum(0) - segment_lens
        permuted_indices = (input_starts[seg_ids] + torch.arange(total, device=device) - output_starts[seg_ids]).long()

        num_tokens_per_expert = t_mat.sum(0)
        return routed_input.shape, routed_input[permuted_indices, :], permuted_indices, num_tokens_per_expert

    def _unpermute(self, routed_output, input_shape, permuted_indices):
        out_unpermuted = routed_output.new_empty(input_shape)
        out_unpermuted[permuted_indices, :] = routed_output
        return out_unpermuted

    def dispatch(self, x: torch.Tensor, top_scores: torch.Tensor, selected_experts_indices: torch.Tensor):
        if self.sp_size > 1:
            x, top_scores, selected_experts_indices = self._split_along_sp(x, top_scores, selected_experts_indices)

        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1).float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts - 1,
        )

        token_indices_experts_sorted = torch.argsort(selected_experts_indices.view(-1), stable=True)
        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k
        routed_input = x[token_indices_experts_sorted]

        if self.score_before_experts:
            routed_input = (routed_input.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)).to(x.dtype)

        # Skip all-to-all logic entirely if ep_mesh is missing (EP=1)
        if self.ep_mesh is None:
            metadata = AllToAllDispatchMetadata(
                token_indices_experts_sorted, top_scores_experts_sorted, 
                None, None, None, None
            )
            return routed_input, num_tokens_per_expert, metadata

        ep_size = self.ep_mesh.size()
        num_tokens_per_expert_group = torch.empty_like(num_tokens_per_expert).repeat(ep_size)
        dist.all_to_all_single(num_tokens_per_expert_group, num_tokens_per_expert, group=self.ep_mesh.get_group())

        input_splits = num_tokens_per_expert.view(ep_size, -1).sum(dim=1).cpu().tolist()
        output_splits = num_tokens_per_expert_group.view(ep_size, -1).sum(dim=1).cpu().tolist()

        routed_input = all_to_all_single_autograd(routed_input, output_splits, input_splits, self.ep_mesh)

        num_local_experts = num_tokens_per_expert_group.shape[0] // ep_size
        input_shape, routed_input, permuted_indices, num_tokens_per_expert_group = self._permute(
            routed_input, num_tokens_per_expert_group, ep_size, num_local_experts
        )

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted, top_scores_experts_sorted, 
            input_shape, permuted_indices, input_splits, output_splits
        )
        return routed_input, num_tokens_per_expert_group, metadata

    def combine(self, routed_output: torch.Tensor, metadata: "AllToAllDispatchMetadata", x: torch.Tensor, shared_experts: torch.nn.Module | None = None) -> torch.Tensor:
        if self.ep_mesh is not None:
            routed_output = self._unpermute(routed_output, metadata.input_shape, metadata.permuted_indices)
            routed_output = all_to_all_single_autograd(routed_output, metadata.input_splits, metadata.output_splits, self.ep_mesh)

        out = shared_experts(x) if shared_experts is not None else torch.zeros_like(x)

        if not self.score_before_experts:
            routed_output = (routed_output.to(torch.float32) * metadata.top_scores_experts_sorted.reshape(-1, 1)).to(routed_output.dtype)

        token_indices_experts_sorted = metadata.token_indices_experts_sorted
        if self.sp_size > 1:
            local_num_tokens = x.shape[0] // self.sp_size
            token_indices_experts_sorted = token_indices_experts_sorted + local_num_tokens * self.sp_rank

        out.scatter_add_(0, token_indices_experts_sorted.reshape(-1, 1).expand(-1, x.shape[-1]), routed_output)
        return out
