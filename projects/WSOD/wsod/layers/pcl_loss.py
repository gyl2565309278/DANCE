# Copyright (c) Facebook, Inc. and its affiliates.
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn import _reduction as _Reduction

from wsod import _C


class _PCLLoss(Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        target: Tensor,
        weight: Tensor,
        cluster: Tensor,
        pc_input: Tensor,
        ignore_index: int = -1,
        reduction: str = "mean",
    ):
        ctx.save_for_backward(input, target, weight, cluster, pc_input)
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction

        output = _C.pcl_loss_forward(
            input,
            target,
            weight,
            cluster,
            pc_input,
            _Reduction.get_enum(reduction),
            ignore_index,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (input, target, weight, cluster, pc_input) = ctx.saved_tensors
        grad_input = _C.pcl_loss_backward(
            grad_output,
            input,
            target,
            weight,
            cluster,
            pc_input,
            _Reduction.get_enum(ctx.reduction),
            ctx.ignore_index,
        )
        return grad_input, None, None, None, None, None, None


pcl_loss = _PCLLoss.apply


class PCLLoss(nn.Module):
    def __init__(self, ignore_index: int = -1, reduction: str = "mean"):
        r"""
        Args:
            ignore_index (int, optional): Specifies a target value that is ignored
                and does not contribute to the input gradient. When :attr:`size_average` is
                ``True``, the loss is averaged over non-ignored targets. Note that
                :attr:`ignore_index` is only applicable when the target contains class indices.
            reduction (str, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
                be applied, ``'mean'``: the weighted mean of the output is taken,
                ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in
                the meantime, specifying either of those two args will override
                :attr:`reduction`. Default: ``'mean'``
        """
        super(PCLLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor, weight: Tensor, cluster: Tensor, pc_input: Tensor):
        return pcl_loss(
            input,
            target,
            weight,
            cluster,
            pc_input,
            self.ignore_index,
            self.reduction,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "ignore_index=" + str(self.ignore_index)
        tmpstr += ", reduction=" + str(self.reduction)
        tmpstr += ")"
        return tmpstr
