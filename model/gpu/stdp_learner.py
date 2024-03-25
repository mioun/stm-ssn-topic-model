from typing import Callable, Union

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from snntorch.functional import probe


def stdp_linear_single_step(
        fc: nn.Linear, in_spike: torch.Tensor, out_spike: torch.Tensor,
        trace_pre: Union[float, torch.Tensor, None],
        trace_post: Union[float, torch.Tensor, None],
        tau_pre: float, tau_post: float,
        f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
    if trace_pre is None:
        trace_pre = 0.

    if trace_post is None:
        trace_post = 0.

    weight = fc.weight.data
    trace_pre = torch.clamp(trace_pre - trace_pre / tau_pre + in_spike, 0, 1)  # shape = [batch_size, N_in]
    trace_post = trace_post - trace_post / tau_post + out_spike   # shape = [batch_size, N_out]
    #0.1
    delta_w_post = (out_spike.unsqueeze(-1) * (
            trace_pre.unsqueeze(1) - trace_post.unsqueeze(-1) * f_post(weight).unsqueeze(0))).sum(0)
    return trace_pre, trace_post, delta_w_post


class STDPLearner2(nn.Module):
    def __init__(
            self,
            synapse: Union[nn.Conv2d, nn.Linear], sn,
            tau_pre: float, tau_post: float,
            f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
    ):
        super().__init__()
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = probe.InputMonitor(synapse)
        self.out_spike_monitor = probe.OutputMonitor(sn)
        self.trace_pre = None
        self.trace_post = None

    def reset(self):
       # super(STDPLearner2, self).reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self):
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self):
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    def step(self, on_grad: bool = True, scale: float = 1.):
        length = self.in_spike_monitor.records.__len__()
        delta_w = None

        if isinstance(self.synapse, nn.Linear):
            stdp_f = stdp_linear_single_step
        else:
            raise NotImplementedError(self.synapse)
        for _ in range(length):
            in_spike = self.in_spike_monitor.records.pop(0)  # [batch_size, N_in]
            out_spike = self.out_spike_monitor.records.pop(0)  # [batch_size, N_out]

            self.trace_pre, self.trace_post, dw = stdp_f(
                self.synapse, in_spike, out_spike,
                self.trace_pre, self.trace_post,
                self.tau_pre, self.tau_post,
                self.f_pre, self.f_post
            )
            if scale != 1.:
                dw *= scale

            delta_w = dw if (delta_w is None) else (delta_w + dw)
        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w.half()
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w
