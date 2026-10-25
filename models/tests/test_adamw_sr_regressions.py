"""BF16 optimizer correctness: replica agreement, moment decay, and resume.

These tests run on CPU; the distributed check uses two Gloo/DDP processes.
"""
import torch

from train.adamw_sr import AdamWSR


def test_fp32_moment_decay():
    p = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    opt = AdamWSR([p], lr=0, weight_decay=0, bucket_mb=1)
    p.grad = torch.ones_like(p)
    opt.step()
    initial = opt.state[p]['exp_avg_sq'].clone()
    p.grad.zero_()
    for _ in range(1000):
        opt.step()
    torch.testing.assert_close(opt.state[p]['exp_avg_sq'].float(), initial.float() * .999**1000,
                               rtol=2e-5, atol=0)
    assert opt.state[p]['exp_avg'].dtype == torch.float32
    assert opt.state[p]['exp_avg_sq'].dtype == torch.float32


def test_replicas_ignore_global_rng():
    ps = [torch.nn.Parameter(torch.linspace(.1, 1, 4096).bfloat16()) for _ in range(2)]
    opts = [AdamWSR([p], lr=1e-5, bucket_mb=1) for p in ps]
    for step in range(2000):
        rng_unchanged = []
        for rank, (p, opt) in enumerate(zip(ps, opts)):
            torch.manual_seed(42 + rank + step)
            torch.rand(7 + rank * 13)
            before = torch.get_rng_state().clone()
            p.grad = torch.full_like(p, .01)
            opt.step()
            rng_unchanged.append(torch.equal(before, torch.get_rng_state()))
        assert torch.equal(ps[0], ps[1])
        assert all(rng_unchanged)


def test_bf16_resume_preserves_moments_and_rounding():
    import copy
    p = torch.nn.Parameter(torch.linspace(.1, 1, 256).bfloat16())
    opt = AdamWSR([p], lr=1e-5, bucket_mb=1)
    for _ in range(7):
        p.grad = torch.full_like(p, .013)
        opt.step()
    saved = copy.deepcopy(opt.state_dict())
    q = torch.nn.Parameter(p.detach().clone())
    resumed = AdamWSR([q], lr=1e-5, bucket_mb=1)
    resumed.load_state_dict(saved)
    for name in ('exp_avg', 'exp_avg_sq'):
        assert resumed.state[q][name].dtype == torch.float32
        assert torch.equal(opt.state[p][name], resumed.state[q][name])
    for i in range(20):
        for param, optimizer in ((p, opt), (q, resumed)):
            param.grad = torch.full_like(param, .01 * (i + 1))
            optimizer.step()
        assert torch.equal(p, q)
    for state in saved['state'].values():
        for name in ('exp_avg', 'exp_avg_sq'):
            state[name] = state[name].bfloat16()
    resumed.load_state_dict(saved)
    assert resumed.state[q]['exp_avg_sq'].dtype == torch.float32


def test_dcp_preserves_moment_precision(tmp_path):
    import torch.distributed.checkpoint as dcp
    p = torch.nn.Parameter(torch.ones(64, dtype=torch.bfloat16))
    opt = AdamWSR([p], bucket_mb=1)
    p.grad = torch.full_like(p, .013)
    opt.step()
    expected = opt.state[p]['exp_avg_sq'].clone()
    dcp.save({'optimizer': opt}, checkpoint_id=tmp_path / 'checkpoint')
    q = torch.nn.Parameter(p.detach().clone())
    resumed = AdamWSR([q], bucket_mb=1)
    q.grad = torch.zeros_like(q)
    resumed.step()
    dcp.load({'optimizer': resumed}, checkpoint_id=tmp_path / 'checkpoint')
    assert resumed.param_groups[0]['step'] == 1
    assert resumed.state[q]['exp_avg_sq'].dtype == torch.float32
    assert torch.equal(expected, resumed.state[q]['exp_avg_sq'])


def _ddp_rounding_worker(rank, init_path):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{init_path}', rank=rank, world_size=2)
    try:
        torch.manual_seed(42 + rank)
        model = DistributedDataParallel(torch.nn.Linear(16, 8, bias=False).bfloat16())
        opt = AdamWSR(model.parameters(), lr=1e-4, bucket_mb=1)
        for _ in range(100):
            opt.zero_grad(set_to_none=True)
            # Distinct local data and RNG consumption, synchronized gradients.
            inputs = torch.randn(4, 16).bfloat16()
            model(inputs).float().square().mean().backward()
            opt.step()
            weight = model.module.weight.detach()
            copies = [torch.empty_like(weight) for _ in range(2)]
            dist.all_gather(copies, weight)
            assert torch.equal(copies[0], copies[1])
    finally:
        dist.destroy_process_group()


def test_two_process_ddp_replica_agreement(tmp_path):
    torch.multiprocessing.spawn(_ddp_rounding_worker,
                                args=(str(tmp_path / 'gloo_init'),), nprocs=2, join=True)


def test_matches_fp32_adamw_reference():
    p = torch.nn.Parameter(torch.linspace(-.1, .1, 64))
    q = torch.nn.Parameter(p.detach().clone())
    ours = AdamWSR([p], lr=1e-3, bucket_mb=1)
    reference = torch.optim.AdamW([q], lr=1e-3, foreach=False)
    for step in range(50):
        grad = torch.linspace(-.01, .02, 64) * ((step % 3) - 1)
        p.grad = grad.clone()
        q.grad = grad.clone()
        ours.step()
        reference.step()
    torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
    for name in ('exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(ours.state[p][name], reference.state[q][name])


def test_rounding_replicas_with_multiple_buckets_and_groups():
    def make_replica():
        parameters = [torch.nn.Parameter(torch.full((600_000,), .1, dtype=torch.bfloat16)),
                      torch.nn.Parameter(torch.full((64,), .2, dtype=torch.bfloat16))]
        optimizer = AdamWSR([{'params': [parameters[0]], 'lr': 1e-4},
                             {'params': [parameters[1]], 'lr': 2e-4}], bucket_mb=1)
        return parameters, optimizer

    replicas = [make_replica(), make_replica()]
    for step in range(3):
        for rank, (parameters, optimizer) in enumerate(replicas):
            torch.manual_seed(42 + rank + step)
            for p in parameters:
                p.grad = torch.full_like(p, .013)
            optimizer.step()
        for a, b in zip(replicas[0][0], replicas[1][0]):
            assert torch.equal(a, b)
    # Different buckets should not reuse the same random-stream prefix.
    first = replicas[0][0][0].detach()
    assert not torch.equal(first[:75_712], first[524_288:])
