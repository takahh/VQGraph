import torch
import torch.distributed as distributed
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack
from torch import nn, einsum
from torch.ao.quantization import quantize
from torch.amp import autocast
from torch.onnx.symbolic_opset9 import pairwise_distance
from einops import rearrange, repeat
from torch.distributions import MultivariateNormal


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def noop(*args, **kwargs):
    pass


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    if temperature == 0:
        return t.argmax(dim=dim)

    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0)


def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]


def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)


def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)


def all_gather_variably_sized(x, sizes, dim=0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src=i, async_op=True)
        all_x.append(t)

    distributed.barrier()
    return all_x


def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, '1 ... -> ...')

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim=0)
    print(f"{local_samples} local_samples")
    print(f"{num} num")

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src=0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim=0)
    out = torch.cat(all_samples, dim=0)

    return rearrange(out, '... -> 1 ...')


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


# data,
# self.codebook_size,
# self.kmeans_iters,
# # use_cosine_sim=True,
# sample_fn = self.sample_fn,
# all_reduce_fn = self.kmeans_all_reduce_fn
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def gmm(
        samples,
        cluster_size=500,  # Enforced fixed cluster size
        num_iters=50,
        sample_fn=None,  # Optional sampling function
        all_reduce_fn=lambda x: x  # No-op by default
):
    """
    Gaussian Mixture Model (GMM) with a fixed number of bins (500 clusters).

    Args:
        samples: Tensor of shape [num_codebooks, num_samples, dim].
        cluster_size: Fixed cluster size (default=500).
        num_iters: Number of EM iterations.
        sample_fn: Optional sampling function.
        all_reduce_fn: Optional reduction function for distributed training.

    Returns:
        bins: Tensor containing cluster assignments for each sample.
    """
    num_codebooks, num_samples, dim = samples.shape
    num_clusters = cluster_size  # Fixed bins/clusters

    # Initialize means using k-means++ logic
    means = torch.empty(num_codebooks, num_clusters, dim, dtype=samples.dtype, device=samples.device)
    for h in range(num_codebooks):
        means[h, 0] = samples[h][torch.randint(0, num_samples, (1,))]
        for k in range(1, num_clusters):
            dists = torch.cdist(samples[h], means[h, :k], p=2) ** 2
            min_dists, _ = torch.min(dists, dim=1)
            prob = min_dists / min_dists.sum()
            chosen_idx = torch.multinomial(prob, 1)
            means[h, k] = samples[h, chosen_idx]

    # Initialize covariances and weights
    covariances = torch.eye(dim, device=samples.device).unsqueeze(0).repeat(num_codebooks, num_clusters, 1, 1)
    weights = torch.ones(num_codebooks, num_clusters, device=samples.device) / num_clusters

    for _ in range(num_iters):
        # E-step: Compute responsibilities
        responsibilities = torch.zeros(num_codebooks, num_samples, num_clusters, device=samples.device)
        for h in range(num_codebooks):
            for k in range(num_clusters):
                mvn = MultivariateNormal(means[h, k], covariance_matrix=covariances[h, k])
                responsibilities[h, :, k] = weights[h, k] * mvn.log_prob(samples[h]).exp()

        responsibilities_sum = responsibilities.sum(dim=-1, keepdim=True)
        responsibilities = responsibilities / responsibilities_sum.clamp(min=1e-9)  # Normalize responsibilities safely

        # M-step: Update means, covariances, and weights
        for h in range(num_codebooks):
            for k in range(num_clusters):
                resp_k = responsibilities[h, :, k]
                N_k = resp_k.sum()
                if N_k > 0:
                    means[h, k] = (resp_k.unsqueeze(-1) * samples[h]).sum(dim=0) / N_k
                    diff = samples[h] - means[h, k]
                    covariances[h, k] = (resp_k.unsqueeze(-1).unsqueeze(-1) *
                                         (diff.unsqueeze(-1) * diff.unsqueeze(-2))).sum(dim=0) / N_k
                    weights[h, k] = N_k / num_samples

        all_reduce_fn(means)

    # Compute final bins (assignments) for a fixed cluster size
    bins = torch.argmax(responsibilities, dim=-1)
    return means, bins


import torch
from einops import rearrange, repeat


def kmeans(
        samples,
        num_clusters,
        num_iters=100,
        use_cosine_sim=False,
        sample_fn=batched_sample_vectors,
        all_reduce_fn=noop
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    num_iters = 50
    means = sample_fn(samples, num_clusters)
    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -torch.cdist(samples, means, p=2)

        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)

        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins


def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b n -> h b n d', d=dim)
    embeds = repeat(embeds, 'h c d -> h b c d', b=batch)
    return embeds.gather(2, indices)


import torch


def compute_contrastive_loss(z, atom_types, margin=1.0):
    """
    Contrastive loss to separate different atom types.
    """
    # Compute pairwise distances
    pairwise_distances = torch.cdist(z, z, p=2)  # Pairwise Euclidean distances

    # Create a mask for same atom types
    same_type_mask = (atom_types[:, None] == atom_types[None, :]).float()  # Mask for same atom type

    # Compute positive and negative losses
    positive_loss = same_type_mask * pairwise_distances ** 2  # Pull same types together
    negative_loss = (1.0 - same_type_mask) * torch.clamp(margin - pairwise_distances,
                                                         min=0.0) ** 2  # Push apart different types
    # Combine and return mean loss
    return (positive_loss + negative_loss).mean()


def feat_elem_divergence_loss(embed_ind, atom_types, num_codebooks=1500, temperature=0.02, normalize="frobenius", alpha=1.0):
    device = embed_ind.device

    # Validate embed_ind
    # print(f"embed_ind: min={embed_ind.min().item()}, max={embed_ind.max().item()}, shape={embed_ind.shape}")
    # print(embed_ind[:30])
    embed_ind = torch.clamp(embed_ind, min=0, max=num_codebooks - 1)
    embed_ind = embed_ind.long()

    # Validate atom_types
    atom_types = torch.nan_to_num(atom_types, nan=0.0, posinf=1.0, neginf=-1.0)
    # print(f"atom_types: min={atom_types.min().item()}, max={atom_types.max().item()}, shape={atom_types.shape}")
    assert torch.isfinite(atom_types).all(), "atom_types contains NaNs or Inf values!"

    # Map atom_types to sequential indices
    unique_atom_numbers = torch.unique(atom_types).tolist()
    atom_number_to_index = {atom: idx for idx, atom in enumerate(unique_atom_numbers)}
    atom_types_mapped = torch.tensor([atom_number_to_index[atom] for atom in atom_types.tolist()], device=device)

    # Create one-hot representations
    embed_one_hot = torch.nn.functional.one_hot(embed_ind, num_classes=num_codebooks).float().to(device)
    atom_type_one_hot = torch.nn.functional.one_hot(atom_types_mapped, num_classes=len(unique_atom_numbers)).float().to(device)

    # Stabilize embed_one_hot
    embed_one_hot = embed_one_hot - embed_one_hot.max(dim=-1, keepdim=True).values

    # Compute soft assignments
    soft_assignments = torch.softmax(embed_one_hot / temperature, dim=-1)

    # Ensure soft_assignments is 2D
    soft_assignments = soft_assignments.view(-1, soft_assignments.shape[-1])
    atom_type_one_hot = atom_type_one_hot.view(-1, atom_type_one_hot.shape[-1])

    # Compute co-occurrence matrix
    co_occurrence = torch.einsum("ni,nj->ij", [soft_assignments, atom_type_one_hot])
    co_occurrence_normalized = co_occurrence / (co_occurrence.sum(dim=1, keepdim=True) + 1e-6)

    # Compute row-wise entropy
    row_entropy = -torch.sum(co_occurrence_normalized * torch.log(co_occurrence_normalized + 1e-6), dim=1)

    # Loss: Average entropy across all rows
    sparsity_loss = row_entropy.mean()

    return sparsity_loss



import torch.nn.functional as F
def increase_non_empty_clusters(embed_ind, embeddings, num_clusters, target_non_empty_clusters=500, min_cluster_size=5):
    # Count the size of each cluster
    cluster_sizes = [(k, (embed_ind == k).sum().item()) for k in range(num_clusters)]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)  # Sort by size (descending)

    # Identify non-empty clusters
    non_empty_clusters = [k for k, size in cluster_sizes if size > 0]
    current_non_empty_count = len(non_empty_clusters)

    if current_non_empty_count < target_non_empty_clusters:
        # print(f"Increasing clusters from {current_non_empty_count} to {target_non_empty_clusters}...")
        new_embed_ind = embed_ind.clone()

        # Determine how many clusters need to be added
        clusters_to_add = target_non_empty_clusters - current_non_empty_count

        # Start splitting the largest clusters
        for k, size in cluster_sizes:
            if clusters_to_add == 0:
                break
            if size > min_cluster_size:  # Only split clusters with enough points
                cluster_mask = (embed_ind == k)
                cluster_points = embeddings[cluster_mask]

                # Split cluster into two
                midpoint = cluster_points.mean(dim=0)
                distances = torch.norm(cluster_points - midpoint, dim=1)
                split_mask = distances > distances.median()  # Split into two groups

                # Create a new cluster for the split points
                new_cluster_id = embed_ind.max().item() + 1  # Ensure unique cluster ID
                new_embed_ind[cluster_mask] = torch.where(
                    split_mask,
                    torch.tensor(new_cluster_id, device=embed_ind.device, dtype=embed_ind.dtype),
                    k
                )
                clusters_to_add -= 1
                # print(f"Created new cluster {new_cluster_id} with size {(new_embed_ind == new_cluster_id).sum().item()}")

        # print(f"Final non-empty clusters: {target_non_empty_clusters - clusters_to_add}")
        return new_embed_ind
    else:
        # print(f"No need to increase clusters; current count is {current_non_empty_count}.")
        return embed_ind

class EuclideanCodebook(nn.Module):
    def __init__(
            self,
            dim,
            codebook_size,
            num_codebooks=1,
            kmeans_init=True,
            kmeans_iters=300,
            sync_kmeans=True,
            decay=0.1,
            eps=1e-5,
            threshold_ema_dead_code=2,
            use_ddp=False,
            learnable_codebook=False,
            sample_codebook_temp=0
    ):
        # print(f"Using EuclidCodebook {num_codebooks}")
        super().__init__()
        self.decay = decay
        init_fn = uniform_init if not kmeans_init else torch.zeros
        # print(f"init_fn: {init_fn}")
        embed = init_fn(num_codebooks, codebook_size, dim)
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        assert not (
                    use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init is not compatible with multiple codebooks in distributed environment for now'
        # use_ddp = True
        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)


    def reset_kmeans(self):
        self.initted.data.copy_(torch.Tensor([False]))


    @torch.jit.ignore
    def init_embed_(self, data):
        # if self.initted:
        #     return
        # embed, cluster_size = gmm(
        #     data,
        #     self.codebook_size,
        #     self.kmeans_iters,
        #     # use_cosine_sim=True,
        #     sample_fn=self.sample_fn,
        #     all_reduce_fn=self.kmeans_all_reduce_fn
        # )

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn
        )
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

        return embed

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0))):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            self.embed.data[ind][mask] = rearrange(sampled, '1 ... -> ...')

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask=expired_codes)

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        needs_codebook_dim = x.ndim < 4
        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, 'h ... d -> h (...) d')
        # -----------------------------------------------------------------------------
        # run simple k-means to set initial codebook
        # -----------------------------------------------------------------------------
        # if self.training:
        #     self.init_embed_(flatten)
        # -----------------------------------------------------------------------------
        # prepare for updating centroids
        # -----------------------------------------------------------------------------
        embed = self.embed if self.learnable_codebook else self.embed.detach()
        init_cb = self.embed.detach().clone().contiguous()
        dist = -torch.cdist(flatten, embed, p=2)
        embed_ind = gumbel_sample(dist, dim=-1, temperature=self.sample_codebook_temp)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = batched_embedding(embed_ind, self.embed)
        # -----------------------------------------------------------------------------
        # Update centroids (in an ML friendly way)
        # -----------------------------------------------------------------------------
        if self.training:
            cluster_size = embed_onehot.sum(dim=1)
            self.all_reduce_fn(cluster_size)
            self.cluster_size.data.lerp_(cluster_size, 1 - self.decay)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            self.all_reduce_fn(embed_sum.contiguous())
            self.embed_avg.data.lerp_(embed_sum, 1 - self.decay)

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            self.embed = torch.nn.Parameter(embed_normalized)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))
            # quantize, embed_ind, dist, embed, latents
        return quantize, embed_ind, dist, self.embed, flatten, init_cb  # flatten と x はどう違うのか？commitment loss には x をしよう。


class CosineSimCodebook(nn.Module):
    def __init__(
            self,
            dim,
            codebook_size,
            num_codebooks=1,
            kmeans_init=False,
            kmeans_iters=10,
            sync_kmeans=True,
            decay=0.8,
            eps=1e-5,
            threshold_ema_dead_code=0,
            use_ddp=False,
            learnable_codebook=False,
            sample_codebook_temp=0.
    ):
        super().__init__()
        self.decay = decay

        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

    @torch.jit.ignore
    # @torch.jit.unused
    def init_embed_(self, data):
        if self.initted:
            return
        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim=True,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn
        )
        self.embed.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        # this line means init_embed is run only once
        self.initted.data.copy_(torch.Tensor([True]))


    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)

        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0))):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            self.embed.data[ind][mask] = rearrange(sampled, '1 ... -> ...')

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask=expired_codes)


    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        needs_codebook_dim = x.ndim < 4
        x = x.float()
        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, 'h ... d -> h (...) d')
        flatten = l2norm(flatten)
        # -----------------------
        # initialize codebook
        # optimization of codebook done here
        # -----------------------
        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        embed = l2norm(embed)
        dist = einsum('h n d, h c d -> h n c', flatten, embed)
        embed_ind = gumbel_sample(dist, dim=-1, temperature=self.sample_codebook_temp)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = batched_embedding(embed_ind, self.embed)

        if self.training:
            bins = embed_onehot.sum(dim=1)
            self.all_reduce_fn(bins)

            self.cluster_size.data.lerp_(bins, 1 - self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            self.all_reduce_fn(embed_sum)

            embed_normalized = embed_sum / rearrange(bins, '... -> ... 1')
            embed_normalized = l2norm(embed_normalized)

            embed_normalized = torch.where(
                rearrange(zero_mask, '... -> ... 1'),
                embed,
                embed_normalized
            )

            self.embed.data.lerp_(embed_normalized, 1 - self.decay)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        return quantize, embed_ind, dist, self.embed, flatten

    # main class


class VectorQuantize(nn.Module):
    def __init__(
            self,
            dim,
            codebook_size,
            codebook_dim=None,
            heads=1,
            separate_codebook_per_head=False,
            decay=0.8,
            eps=1e-5,
            kmeans_init=True,
            kmeans_iters=600,
            sync_kmeans=True,
            use_cosine_sim=False,
            threshold_ema_dead_code=0,
            channel_last=True,
            accept_image_fmap=False,
            commitment_weight=0.003,
            margin_weight=0.8,
            spread_weight=0.2,
            pair_weight=0.01,
            lamb_div_ele=1,
            lamb_div_bonds=1,
            lamb_div_aroma=1,
            lamb_div_ringy=1,
            lamb_div_h_num=1,
            lamb_sil=1,
            orthogonal_reg_active_codes_only=False,
            orthogonal_reg_max_codes=None,
            sample_codebook_temp=0.,
            sync_codebook=False
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = default(codebook_dim, dim)  # use coocbook_dim if not None
        codebook_input_dim = codebook_dim * heads
        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.eps = eps
        self.commitment_weight = commitment_weight

        has_codebook_orthogonal_loss = margin_weight > 0

        self.margin_weight = margin_weight
        self.spread_weight = spread_weight
        self.lamb_div_ele = lamb_div_ele
        self.lamb_div_bonds = lamb_div_bonds
        self.lamb_div_aroma = lamb_div_aroma
        self.lamb_div_ringy = lamb_div_ringy
        self.lamb_div_h_num = lamb_div_h_num
        self.lamb_sil = lamb_sil
        self.pair_weight = pair_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook
        self._codebook = codebook_class(
            dim=codebook_dim,
            num_codebooks=heads if separate_codebook_per_head else 1,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sync_kmeans=sync_kmeans,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_ddp=sync_codebook,
            learnable_codebook=has_codebook_orthogonal_loss,
            sample_codebook_temp=sample_codebook_temp
        )

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self):
        codebook = self._codebook.embed
        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, '1 ... -> ...')

    def get_codes_from_indices(self, indices):
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, '... h d -> ... (h d)')

        indices, ps = pack([indices], 'b * h')
        indices = rearrange(indices, 'b n h -> b h n')

        indices = repeat(indices, 'b h n -> b h n d', d=codebook.shape[-1])
        codebook = repeat(codebook, 'h n d -> b h n d', b=indices.shape[0])

        codes = codebook.gather(2, indices)
        codes = rearrange(codes, 'b h n d -> b n (h d)')
        codes, = unpack(codes, ps, 'b * d')
        return codes


    def compute_contrastive_loss(self, z, atom_types, margin=1.0):
        """
        Contrastive loss to separate different atom types.
        """
        # Compute pairwise distances
        pairwise_distances = torch.cdist(z, z, p=2)  # Pairwise Euclidean distances

        # Mask the diagonal (distances to itself)
        mask = ~torch.eye(pairwise_distances.size(0), dtype=torch.bool)
        non_diag_distances = pairwise_distances[mask]

        # Compute mean, max, and min distances
        mean_distance = non_diag_distances.mean().item()
        max_distance = non_diag_distances.max().item()
        min_distance = non_diag_distances.min().item()

        # Print results
        print(f"Mean distance: {mean_distance:.4f}")
        print(f"Max distance: {max_distance:.4f}")
        print(f"Min distance: {min_distance:.4f}")

        # Create a mask for same atom types
        same_type_mask = (atom_types[:, None] == atom_types[None, :]).float()  # Mask for same atom type

        # Compute positive and negative losses
        positive_loss = same_type_mask * pairwise_distances ** 2  # Pull same types together
        negative_loss = (1.0 - same_type_mask) * torch.clamp(margin - pairwise_distances,
                                                             min=0.0) ** 2  # Push apart different types

        # Combine and return mean loss
        return (positive_loss + negative_loss).mean()


    def fast_silhouette_loss(self, embeddings, embed_ind, num_clusters, target_non_empty_clusters=500):
        # Preprocess clusters to ensure the desired number of non-empty clusters
        embed_ind = increase_non_empty_clusters(embed_ind, embeddings, num_clusters, target_non_empty_clusters)
        embed_ind.data.copy_(embed_ind)

        # Compute pairwise distances for all points
        pairwise_distances = torch.cdist(embeddings, embeddings)  # Shape: (N, N)

        # Initialize lists to store distances for non-empty clusters
        intra_cluster_distances = []
        inter_cluster_distances = []
        empty_cluster_count = 0  # Counter for empty clusters

        # Iterate over clusters
        for k in range(num_clusters):
            cluster_mask = (embed_ind == k)  # Mask for cluster k
            cluster_indices = cluster_mask.nonzero(as_tuple=True)[0]

            if cluster_indices.numel() == 0:  # If the cluster is empty
                empty_cluster_count += 1  # Increment the empty cluster count
                continue

            # Compute intra-cluster distances
            cluster_distances = pairwise_distances[cluster_indices][:, cluster_indices]
            if cluster_distances.numel() > 1:
                intra_cluster_distances.append(cluster_distances.mean().item())
            else:
                intra_cluster_distances.append(0)

            # Compute inter-cluster distances
            other_mask = ~cluster_mask
            if other_mask.sum() > 0:
                other_distances = pairwise_distances[cluster_indices][:, other_mask]
                inter_cluster_distances.append(other_distances.mean().item())
            else:
                inter_cluster_distances.append(float('inf'))

        # Print the number of empty clusters
        # print(f"Number of empty clusters: {empty_cluster_count}")

        # Convert intra- and inter-cluster distances to tensors
        a = torch.tensor(intra_cluster_distances, device=embeddings.device)
        b = torch.tensor(inter_cluster_distances, device=embeddings.device)

        # Compute silhouette coefficients
        epsilon = 1e-6  # To avoid division by zero
        silhouette_coefficients = (b - a) / torch.max(a + epsilon, b + epsilon)
        silhouette_coefficients = torch.nan_to_num(silhouette_coefficients, nan=0.0)

        # Return the mean silhouette loss
        return embed_ind, -silhouette_coefficients.mean()


    def orthogonal_loss_fn(self, embed_ind, t, init_feat, latents, min_distance=0.5):
        # Normalize embeddings (optional: remove if not necessary)
        t_norm = torch.norm(t, dim=1, keepdim=True) + 1e-6
        t = t / t_norm

        latents_norm = torch.norm(latents, dim=1, keepdim=True) + 1e-6
        latents = latents / latents_norm

        # Pairwise distances
        dist_matrix = torch.squeeze(torch.cdist(t, t, p=2) + 1e-6)  # Avoid zero distances

        # Remove diagonal
        mask = ~torch.eye(dist_matrix.size(0), dtype=bool, device=dist_matrix.device)
        dist_matrix_no_diag = dist_matrix[mask].view(dist_matrix.size(0), -1)

        # Debug: Log distance statistics
        # print(f"Min: {dist_matrix_no_diag.min().item()}, Max: {dist_matrix_no_diag.max().item()}, Mean: {dist_matrix_no_diag.mean().item()}")

        # Margin loss: Encourage distances >= min_distance
        smooth_penalty = torch.nn.functional.relu(min_distance - dist_matrix_no_diag)
        margin_loss = torch.mean(smooth_penalty)  # Use mean for better gradient scaling

        # Spread loss: Encourage diversity
        spread_loss = torch.var(t)

        # Pair distance loss: Regularize distances
        pair_distance_loss = torch.mean(torch.log(dist_matrix_no_diag))

        # sil loss
        embed_ind_for_sil = torch.squeeze(embed_ind)
        latents_for_sil = torch.squeeze(latents)
        embed_ind, sil_loss = self.fast_silhouette_loss(latents_for_sil, embed_ind_for_sil, t.shape[-2], t.shape[-2])

        # ---------------------------------------------------------------
        # loss to assign different codes for different chemical elements
        # ---------------------------------------------------------------
        atom_type_div_loss = feat_elem_divergence_loss(embed_ind, init_feat[:, 0]).clone().detach()
        atom_type_div_loss = atom_type_div_loss + compute_contrastive_loss(latents, embed_ind)
        bond_num_div_loss = feat_elem_divergence_loss(embed_ind, init_feat[:, 1]).clone().detach()
        aroma_div_loss = feat_elem_divergence_loss(embed_ind, init_feat[:, 4]).clone().detach()
        ringy_div_loss = feat_elem_divergence_loss(embed_ind, init_feat[:, 5]).clone().detach()
        h_num_div_loss = feat_elem_divergence_loss(embed_ind, init_feat[:, 6]).clone().detach()

        # bond_num_div_loss = torch.tensor(feat_elem_divergence_loss(embed_ind, init_feat[:, 1]))
        # aroma_div_loss = torch.tensor(feat_elem_divergence_loss(embed_ind, init_feat[:, 4]))
        # ringy_div_loss = torch.tensor(feat_elem_divergence_loss(embed_ind, init_feat[:, 5]))
        # h_num_div_loss = torch.tensor(feat_elem_divergence_loss(embed_ind, init_feat[:, 6]))

        return (margin_loss, spread_loss, pair_distance_loss, atom_type_div_loss, bond_num_div_loss, aroma_div_loss,
                ringy_div_loss, h_num_div_loss, sil_loss, embed_ind)

    def forward(
            self,
            x,
            init_feat,
            mask=None
    ):
        only_one = x.ndim == 2

        if only_one:
            x = rearrange(x, 'b d -> b 1 d')
        shape, device, heads, is_multiheaded, codebook_size = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size

        need_transpose = not self.channel_last and not self.accept_image_fmap

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        x = self.project_in(x)
        if is_multiheaded:
            ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
            x = rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h=heads)
        # --------------------------------------------------
        # quantize here
        # --------------------------------------------------
        # quantize, embed_ind, dist, self.embed, flatten, init_cb
        quantize, embed_ind, dist, embed, latents, init_cb = self._codebook(x)
        # quantize　: 各データに対応する codebook vector
        # embed_ind : 各データに対応する codebook vector のインデックス
        # dist      : codebook の距離行列
        # embed     : codebook  ← これをプロットに使いたい
        # latents   : 潜在変数ベクトル

        codes = self.get_codes_from_indices(embed_ind)
        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.], device=device, requires_grad=self.training)
        # --------------------------------------------------
        # calculate loss about codebook itself in training
        # --------------------------------------------------
        raw_commit_loss = torch.tensor([0.], device=device, requires_grad=self.training)
        margin_loss = torch.tensor([0.], device=device, requires_grad=self.training)
        spread_loss = torch.tensor([0.], device=device, requires_grad=self.training)
        div_ele_loss = torch.tensor([0.], device=device, requires_grad=self.training)
        pair_distance_loss = torch.tensor([0.], device=device, requires_grad=self.training)
        detached_quantize = torch.tensor([0.], device=device, requires_grad=self.training)
        bond_num_div_loss = torch.tensor([0.], device=device, requires_grad=self.training)
        aroma_div_loss = torch.tensor([0.], device=device, requires_grad=self.training)
        ringy_div_loss = torch.tensor([0.], device=device, requires_grad=self.training)
        h_num_div_loss = torch.tensor([0.], device=device, requires_grad=self.training)
        silh_loss = torch.tensor([0.], device=device, requires_grad=self.training)
        # if self.training:
        if self.commitment_weight > 0:  # 0.25 is assigned
            detached_quantize = quantize.detach()

            if exists(mask):
                # with variable lengthed sequences
                commit_loss = F.mse_loss(detached_quantize, x, reduction='none')

                if is_multiheaded:
                    mask = repeat(mask, 'b n -> c (b h) n', c=commit_loss.shape[0],
                                  h=commit_loss.shape[1] // mask.shape[0])

                commit_loss = commit_loss[mask].mean()
            else:
                commit_loss = F.mse_loss(detached_quantize, x)
            raw_commit_loss = commit_loss
            loss = loss + commit_loss * self.commitment_weight

        # if self.margin_weight > 0:  # now skip because it is zero
        codebook = self._codebook.embed

        if self.orthogonal_reg_active_codes_only:
            # only calculate orthogonal loss for the activated codes for this batch
            unique_code_ids = torch.unique(embed_ind)
            codebook = torch.squeeze(codebook)
            codebook = codebook[unique_code_ids]

        num_codes = codebook.shape[0]
        if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
            rand_ids = torch.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]
            codebook = codebook[rand_ids]
        # ---------------------------------
        # Calculate Codebook Losses
        # ---------------------------------
        # print(f"embed_ind.shape {embed_ind.shape} befpre ")
        (margin_loss, spread_loss, pair_distance_loss, div_ele_loss, bond_num_div_loss, aroma_div_loss,
         ringy_div_loss, h_num_div_loss, silh_loss, embed_ind) = self.orthogonal_loss_fn(embed_ind, codebook, init_feat, latents)
        # margin_loss, spread_loss = orthogonal_loss_fn(codebook)
        # print(f"embed_ind.shape {embed_ind.shape} after ")
        if embed_ind.ndim == 2:
            embed_ind = rearrange(embed_ind, 'b 1 -> b')  # Reduce if 2D with shape [b, 1]
        elif embed_ind.ndim == 1:
            embed_ind = embed_ind  # Leave as is if already 1D
        else:
            raise ValueError(f"Unexpected shape for embed_ind: {embed_ind.shape}")

        # ---------------------------------
        # Calculate silouhette Losses
        # ---------------------------------
        # silh_loss = silhouette_loss(latents, embed_ind, codebook.shape[0])

        # ---------------------------------
        # linearly combine losses !!!!
        # ---------------------------------
        # loss = (loss + self.lamb_div_ele * div_ele_loss + self.lamb_div_aroma * aroma_div_loss
        #  + self.lamb_div_bonds * bond_num_div_loss + self.lamb_div_aroma * aroma_div_loss
        #  + self.lamb_div_ringy * ringy_div_loss + self.lamb_div_h_num * h_num_div_loss)
        loss = loss + self.lamb_div_ele * div_ele_loss
        # loss = (loss + margin_loss * self.margin_weight + pair_distance_loss * self.pair_weight +
        #         self.spread_weight * spread_loss + self.lamb_sil * silh_loss)
        loss = loss + self.lamb_sil * silh_loss

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, 'h b n d -> b n (h d)', h=heads)
                embed_ind = rearrange(embed_ind, 'h b n -> b n h', h=heads)
            else:
                quantize = rearrange(quantize, '1 (b h) n d -> b n (h d)', h=heads)
                embed_ind = rearrange(embed_ind, '1 (b h) n -> b n h', h=heads)

        quantize = self.project_out(quantize)

        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=height, w=width)
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h=height, w=width)

        if only_one:
            quantize = rearrange(quantize, 'b 1 d -> b d')
            if len(embed_ind.shape) == 2:
                embed_ind = rearrange(embed_ind, 'b 1 -> b')
        #
        # quantized, _, commit_loss, dist, codebook, raw_commit_loss, latents, margin_loss, spread_loss, pair_loss, detached_quantize, x, init_cb
        return (quantize, embed_ind, loss, dist, embed, raw_commit_loss, latents, margin_loss, spread_loss,
                pair_distance_loss, detached_quantize, x, init_cb, div_ele_loss, bond_num_div_loss, aroma_div_loss, ringy_div_loss, h_num_div_loss, silh_loss)
