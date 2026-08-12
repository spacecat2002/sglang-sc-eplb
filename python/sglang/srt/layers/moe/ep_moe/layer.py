from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_dp_global_num_tokens,
    get_is_extend_in_batch,
    set_is_extend_in_batch,
)
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    moe_forward_piecewise_cuda_graph_impl,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPNormalCombineInput,
)
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TopKOutput,
    TopKOutputChecker,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.utils import get_bool_env_var, is_hip, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        DispatchOutput,
    )

_is_hip = is_hip()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


logger = logging.getLogger(__name__)
_expert_load_logging_enabled = get_bool_env_var("SGLANG_ENABLE_EXPERT_LOAD_LOGGING")


def _maybe_log_expert_load(moe_layer, dispatch_output) -> None:
    if not _expert_load_logging_enabled:
        return
    from sglang.srt.layers.moe.expert_load_logger import ExpertLoadLogger
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

    if not DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
        return

    load_logger = ExpertLoadLogger.get()
    load_logger.init_metadata(
        num_local_experts=moe_layer.num_local_experts,
        ep_rank=moe_layer.moe_ep_rank,
        ep_size=moe_layer.moe_ep_size,
        global_rank=(
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        ),
    )
    load_logger.record(
        layer_id=moe_layer.layer_id,
        counts=dispatch_output.num_recv_tokens_per_expert,
    )


def _build_force_balanced_topk_ids(
    template_topk_ids: torch.Tensor,
    *,
    num_experts: int,
    ep_size: int,
    global_token_start: int,
) -> torch.Tensor:
    """Build a deterministic rank-first routing pattern for profiling."""
    num_tokens, top_k = template_topk_ids.shape
    if num_tokens == 0 or top_k == 0:
        return torch.empty_like(template_topk_ids)

    local_experts_per_rank = num_experts // ep_size
    rows = torch.arange(
        global_token_start,
        global_token_start + num_tokens,
        device=template_topk_ids.device,
        dtype=torch.int64,
    )
    slots = torch.arange(top_k, device=template_topk_ids.device, dtype=torch.int64)
    assignment = rows.unsqueeze(1).mul(top_k).add(slots)
    destination_rank = assignment.remainder(ep_size)
    local_expert = assignment.div(ep_size, rounding_mode="floor").remainder(
        local_experts_per_rank
    )
    balanced = destination_rank.mul(local_experts_per_rank).add(local_expert)
    balanced = balanced.to(dtype=template_topk_ids.dtype)
    return balanced.masked_fill(template_topk_ids[:, :1] < 0, -1)


class DeepEPMoE(FusedMoE):
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    Mooncake EP shares the same class, as they expose the same interface.
    """

    _has_printed = False
    _has_logged_router_force_balance = False
    _has_logged_router_force_balance_unsupported = False

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )
        self.router_force_balance_enabled = get_bool_env_var(
            "SGLANG_ROUTER_FORCE_BALANCE"
        )
        if self.router_force_balance_enabled and self.num_fused_shared_experts > 0:
            logger.warning(
                "SGLANG_ROUTER_FORCE_BALANCE does not support fused shared experts; "
                "disabling it for layer %s.",
                self.layer_id,
            )
            self.router_force_balance_enabled = False
        if (
            self.router_force_balance_enabled
            and not DeepEPMoE._has_logged_router_force_balance
        ):
            logger.warning(
                "SGLANG_ROUTER_FORCE_BALANCE is enabled. Prefill routed expert IDs "
                "will be replaced with a deterministic balanced pattern."
            )
            DeepEPMoE._has_logged_router_force_balance = True
        is_humming = (
            get_moe_runner_backend().is_humming()
            or get_moe_runner_backend().is_auto()
            and quant_config is not None
            and quant_config.get_name() == "humming"
        )
        if is_humming:
            self.deprecate_flag = True
        elif _use_aiter:
            self.deprecate_flag = True
        elif _is_npu:
            self.deprecate_flag = True
        elif deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and isinstance(
            quant_config, Fp8Config
        ):
            self.deprecate_flag = True
        elif (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
        ):
            self.deprecate_flag = True
        elif (
            get_moe_runner_backend().is_flashinfer_cutedsl()
            and quant_config is not None
            and quant_config.get_name()
            in ("modelopt_fp4", "modelopt_mixed", "nvfp4_online")
        ):
            self.deprecate_flag = True
        elif (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and get_moe_runner_backend().is_deep_gemm()
            and quant_config is not None
            and quant_config.get_name() == "mxfp4"
        ):
            # MXFP4 experts (e.g. Kimi K3) on the DeepGEMM fp8_fp4 W4A8 path:
            # route through the modern FusedMoE runner (Mxfp4MoEMethod.apply).
            self.deprecate_flag = True
        elif (
            quant_config is None
            and self.w13_weight.dtype == torch.bfloat16
            and get_moe_runner_backend().is_deep_gemm()
            and (get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_pplx())
            and not _is_npu
            and not _is_hip
        ):
            assert (
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            ), "Unquantized DeepEP MoE requires DeepGEMM BF16"
            self.deprecate_flag = True
        else:
            self.deprecate_flag = False

        if self.deprecate_flag:
            return

        if isinstance(quant_config, Fp8Config):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.use_fp8_w8a8 = True
            self.fp8_dtype = torch.float8_e4m3fn
            self.use_w4afp8 = False
        elif isinstance(quant_config, W4AFp8Config):
            self.use_w4afp8 = True
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
        else:
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = False
            self.use_block_quant = False

        self.deepep_mode = get_deepep_mode()
        if (
            self.deepep_mode.enable_low_latency()
            and not _is_npu
            and not _is_hip
            and quant_config is not None
        ):
            # AMD HIP and NPU support low_latency DeepEP without DeepGEMM.
            assert (
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            ), f"DeepEP {self.deepep_mode} mode requires deep_gemm"

    def _a2a_forward_with_output_impl(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # eager run under breakable cuda graph
        saved_is_extend_in_batch = get_is_extend_in_batch()
        set_is_extend_in_batch(True)
        try:
            output.copy_(
                self.forward_impl(
                    hidden_states,
                    StandardTopKOutput(topk_weights, topk_ids, router_logits),
                )
            )
        finally:
            set_is_extend_in_batch(saved_is_extend_in_batch)

    def _a2a_forward_capture_stub(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # Capture pass only: record the buffer address, skip the
        # rank-coupled a2a. Warmup and replay run the real body.
        output.zero_()

    a2a_forward_with_output = eager_on_graph(
        True, capture_stub=_a2a_forward_capture_stub
    )(_a2a_forward_with_output_impl)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        # DeepEP NORMAL mode is not capturable; run it as an eager node.
        if is_in_breakable_cuda_graph():
            assert TopKOutputChecker.format_is_standard(
                topk_output
            ), "Only standard topk output is supported for breakable cuda graph"
            output = torch.empty_like(hidden_states)
            self.a2a_forward_with_output(
                hidden_states,
                topk_output.topk_weights,
                topk_output.topk_ids,
                topk_output.router_logits,
                output,
            )
            return output
        if is_in_tc_piecewise_cuda_graph():
            assert TopKOutputChecker.format_is_standard(
                topk_output
            ), "Only standard topk output is supported for piecewise cuda graph"
            return moe_forward_piecewise_cuda_graph_impl(
                hidden_states,
                topk_output.topk_weights,
                topk_output.topk_ids,
                topk_output.router_logits,
                self.layer_id,
            )
        else:
            return self.forward_impl(hidden_states, topk_output)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        topk_output = self._maybe_force_balance_topk(topk_output)

        if self.deprecate_flag:
            return super().forward_impl(
                hidden_states,
                topk_output,
            )

        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )
        combine_input = self.run_moe_core(dispatch_output)
        return self.dispatcher.combine(combine_input=combine_input)

    def _maybe_force_balance_topk(self, topk_output: TopKOutput) -> TopKOutput:
        if not self.router_force_balance_enabled or not get_is_extend_in_batch():
            return topk_output
        if not isinstance(topk_output, StandardTopKOutput):
            if not DeepEPMoE._has_logged_router_force_balance_unsupported:
                logger.warning(
                    "SGLANG_ROUTER_FORCE_BALANCE only supports standard top-k "
                    "outputs; keeping the original routing."
                )
                DeepEPMoE._has_logged_router_force_balance_unsupported = True
            return topk_output

        routed_num_experts = self.num_experts - self.num_fused_shared_experts
        if routed_num_experts <= 0 or routed_num_experts % self.moe_ep_size != 0:
            if not DeepEPMoE._has_logged_router_force_balance_unsupported:
                logger.warning(
                    "SGLANG_ROUTER_FORCE_BALANCE requires routed experts to be "
                    "divisible by EP size; keeping the original routing."
                )
                DeepEPMoE._has_logged_router_force_balance_unsupported = True
            return topk_output

        global_num_tokens = get_dp_global_num_tokens()
        global_token_start = 0
        if global_num_tokens:
            dp_rank = get_attention_dp_rank()
            global_token_start = sum(global_num_tokens[:dp_rank])

        return StandardTopKOutput(
            topk_weights=topk_output.topk_weights,
            topk_ids=_build_force_balanced_topk_ids(
                topk_output.topk_ids,
                num_experts=routed_num_experts,
                ep_size=self.moe_ep_size,
                global_token_start=global_token_start,
            ),
            router_logits=topk_output.router_logits,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        return self.dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

    def run_moe_core(
        self,
        dispatch_output: DispatchOutput,
    ):
        _maybe_log_expert_load(self, dispatch_output)

        if self.deprecate_flag:
            return super().run_moe_core(dispatch_output)

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            if self.quant_config is None:
                raise NotImplementedError(
                    "Unquantized DeepEP MoE currently supports low_latency mode only"
                )
            elif self.use_w4afp8:
                output = self.forward_cutlass_w4afp8(dispatch_output)
            else:
                assert False, "forward_deepgemm_contiguous is deprecated"
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if self.use_w4afp8:
                output = self.forward_cutlass_w4afp8_masked(dispatch_output)
            else:
                assert False, "forward_deepgemm_masked is deprecated"

        combine_input_wrapper = (
            DeepEPNormalCombineInput
            if DispatchOutputChecker.format_is_deepep_normal(dispatch_output)
            else DeepEPLLCombineInput
        )

        return combine_input_wrapper(
            hidden_states=output,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_args: Optional[Dict[str, Any]] = None,
    ):
        return self.dispatcher.combine(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            overlap_args=overlap_args,
        )

    def forward_cutlass_w4afp8(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        assert self.moe_runner_config.activation in ("silu", "situ")
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_normal(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_cutlass_w4afp8_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        assert self.moe_runner_config.activation in ("silu", "situ")
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_ll(
            layer=self,
            dispatch_output=dispatch_output,
        )


def get_moe_impl_class(quant_config: Optional[QuantizationConfig]):
    # [TODO] kk, temporary solution
    if (
        get_moe_a2a_backend().is_mori()
        or get_moe_a2a_backend().is_deepep()
        or get_moe_a2a_backend().is_mooncake()
        or get_moe_a2a_backend().is_nixl()
        or get_moe_a2a_backend().is_pplx()
    ):
        return DeepEPMoE
    return FusedMoE
