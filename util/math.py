import torch


class UpperBoundGrad(torch.autograd.Function):
    """
    Same as `torch.clamp_max`, but with helpful gradient for `inputs > bound`.
    """

    @staticmethod
    def forward(ctx, input, bound: float):
        ctx.save_for_backward(input)
        ctx.bound = bound

        return input.clamp_max(bound)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        pass_through = (input <= ctx.bound) | (grad_output > 0)
        return grad_output * pass_through, None


def upper_bound(input, bound: float):
    """upper_bound"""
    return UpperBoundGrad.apply(input, bound)


class LowerBoundGrad(torch.autograd.Function):
    """
    Same as `torch.clamp_min`, but with helpful gradient for `inputs > bound`.
    """

    @staticmethod
    def forward(ctx, input, bound: float):
        ctx.save_for_backward(input)
        ctx.bound = bound

        return input.clamp_min(bound)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        pass_through = (input >= ctx.bound) | (grad_output < 0)
        return grad_output * pass_through.float(), None


def lower_bound(input, bound: float):
    """lower_bound"""
    return LowerBoundGrad.apply(input, bound)


def bound(input, min, max):
    """bound"""
    return upper_bound(lower_bound(input, min), max)


def bound_sigmoid(input, scale=10):
    """bound_sigmoid"""
    return bound(input, -scale, scale).sigmoid()


def bound_tanh(input, scale=3):
    """bound_tanh"""
    return bound(input, -scale, scale).tanh()
