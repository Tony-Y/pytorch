# Owner(s): ["module: optimizer", "module: LrScheduler" ]
from functools import partial

import torch
from torch.optim import (
    Adafactor,
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    ASGD,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
    SparseAdam,
)
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    load_tests,
    parametrize,
    TestCase,
)

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

class TestTensorLR(TestCase):
    def setUp(self):
        super().setUp()
        self.param = torch.rand(2, 3, requires_grad=True)
        self.param.grad = torch.rand_like(self.param)

    def _tensor(self, value, dim):
        shape = [1] * dim
        x = torch.tensor(value).reshape(shape)
        self.assertEqual(x.dim(), dim)
        return x

    def _tensor_lr(self, lr_dim):
        return self._tensor(0.01, lr_dim)

    def _tensor_betas(self, beta1_dim, beta2_dim):
        beta1 = self._tensor(0.9, beta1_dim)
        beta2 = self._tensor(0.99, beta2_dim)
        return (beta1, beta2)

    @parametrize(
        "OptimClass",
        [
            Adafactor,
            Adadelta,
            Adagrad,
            Adam,
            Adamax,
            AdamW,
            ASGD,
            NAdam,
            RAdam,
            RMSprop,
            Rprop,
            SGD,
            SparseAdam,
        ],
    )
    @parametrize("lr_dim", range(11))
    def test_optimizers_with_tensor_lr(self, OptimClass, lr_dim):
        if OptimClass is SparseAdam:
            self.param.grad = self.param.grad.to_sparse()

        lr = self._tensor_lr(lr_dim)

        optimizer = OptimClass([self.param], lr=lr)
        optimizer.step()

    @parametrize(
        "OptimClass",
        [
            Adam,
            AdamW,
        ],
    )
    @parametrize(
        "beta1_dim,beta2_dim",
        [(0, 0)] + [(0, i) for i in range(1, 11)] + [(i, 0) for i in range(1, 11)],
    )
    def test_adam_with_tensor_betas(self, OptimClass, beta1_dim, beta2_dim):
        betas = self._tensor_betas(beta1_dim, beta2_dim)

        optimizer = OptimClass([self.param], lr=0.01, betas=betas)
        optimizer.step()

    @parametrize(
        "LRClass",
        [
            partial(LambdaLR, lr_lambda=lambda e: e // 10),
            partial(MultiplicativeLR, lr_lambda=lambda e: 0.95),
            partial(StepLR, step_size=30),
            partial(MultiStepLR, milestones=[30, 80]),
            ConstantLR,
            LinearLR,
            partial(ExponentialLR, gamma=0.9),
            PolynomialLR,
            partial(CosineAnnealingLR, T_max=10),
            lambda opt, **kwargs: ChainedScheduler(
                schedulers=[ConstantLR(opt), ConstantLR(opt)], **kwargs
            ),
            lambda opt, **kwargs: SequentialLR(
                opt,
                schedulers=[ConstantLR(opt), ConstantLR(opt)],
                milestones=[2],
                **kwargs,
            ),
            ReduceLROnPlateau,
            partial(CyclicLR, base_lr=0.01, max_lr=0.1),
            partial(OneCycleLR, max_lr=0.01, total_steps=10, anneal_strategy="linear"),
            partial(CosineAnnealingWarmRestarts, T_0=20),
        ],
    )
    @parametrize("lr_dim", range(11))
    def test_lrschedulers_with_tensor_lr(self, LRClass, lr_dim):
        lr = self._tensor_lr(lr_dim)

        optimizer = Adam([self.param], lr=lr)
        scheduler = LRClass(optimizer)
        optimizer.step()
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(0.001)
        else:
            scheduler.step()


instantiate_parametrized_tests(TestTensorLR)


if __name__ == "__main__":
    print("These tests should be run through test/test_optim.py instead")
