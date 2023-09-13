from typing import cast, List
from unittest.mock import patch

import sympy

import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, IRNode, Pointwise
from torch._inductor.utils import IndentedBuffer, sympy_str


def _arg_str(a):
    if isinstance(a, sympy.Expr):
        return "sympy_expr('" + sympy_str(a) + "')"
    return str(a)


class CutlassEVTEpilogueTypeFormatter:
    """
    Replacement for V.KernelFormatterHandler
    """

    def __init__(self, accumulator_node_name, evt_type_name):
        self.accumulator_node_name = accumulator_node_name
        self.output = IndentedBuffer(0)
        self.var_counter = 0
        self.evt_type_name = evt_type_name
        self.aliases = dict()

    @staticmethod
    def ir_to_evt_string(
        template_output_node_name: str,
        evt_type_name: str,
        epilogue_nodes: List[IRNode],
    ):
        formatter = CutlassEVTEpilogueTypeFormatter(
            template_output_node_name, evt_type_name
        )

        with virtualized.V.set_ops_handler(formatter), patch.object(  # type: ignore[call-arg]
            FlexibleLayout, "allow_indexing", True
        ):
            for node in epilogue_nodes:
                if isinstance(node, ComputedBuffer):
                    pnode = node.data
                else:
                    raise RuntimeError(
                        "Epilogue nodes must be Pointwise nodes, wrapped in a named ComputedBuffer"
                    )
                assert isinstance(pnode, Pointwise)
                pnode = cast(Pointwise, pnode)  # make mypy happy
                index = pnode._index(pnode.ranges)
                result = pnode.inner_fn(index)
                # each epilogue node results in a single "using" statement and may refer to the previous steps by name
                formatter.aliases[node.name] = result
            return formatter.getvalue(result)

    def __getattr__(self, name):
        def inner(*args, **kwargs):
            fargs = [_arg_str(a) for a in args]
            fkwargs = {key: _arg_str(a) for key, a in kwargs.items()}
            fn = getattr(self, f"_op_{name}")
            line = fn(*fargs, **fkwargs)
            self.var_counter += 1
            varname = f"EVT_expr_{self.var_counter}"
            # replace line with a new variable name
            self.output.writeline(f"using {varname} = {line};")
            return varname

        if hasattr(self, f"_op_{name}"):
            return inner
        else:
            raise NotImplementedError(name)

    def _op_load(self, name, index_expr):
        if name == self.accumulator_node_name:
            return f"cutlass::epilogue::fusion::Sm90AccFetch /* :={name} (matmul output in accumulator) */"
        elif name in self.aliases:
            return self.aliases[name]
        else:
            return f"cutlass::epilogue::fusion::Sm90SrcFetch /* :={name} */"

    def _op_constant(self, value, dtype):
        if str(dtype) in ("torch.float16", "torch.float32"):
            return f"cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementAcc> /* value={value}, dtype={dtype} */"
        else:
            raise NotImplementedError(f"Unsupported dtype for constant: {dtype}")

    def _cutlass_binary_functional_op(self, op, a, b):
        # see https://github.com/NVIDIA/cutlass/blob/6407bcdf0a24097b7b016ee105937693c62f9923/include/cutlass/functional.h for ops
        return f"cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::{op}, ElementAcc, ElementAcc, RoundStyle>,{a},{b}>"  # noqa: B950

    def _op_mul(self, a, b):
        return self._cutlass_binary_functional_op("multiplies", a, b)

    def _op_ge(self, a, b):
        return self._cutlass_binary_functional_op("greater_equal", a, b)

    def _op_add(self, a, b):
        return self._cutlass_binary_functional_op("plus", a, b)

    def _op_sub(self, a, b):
        return self._cutlass_binary_functional_op("minus", a, b)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise NotImplementedError()

    def getvalue(self, result):
        self.output.writeline(
            f"using {self.evt_type_name} = EVT_expr_{self.var_counter};"
        )
        return self.output.getvalue()


class CutlassEVTEpilogueArgumentFormatter:
    """
    Replacement for V.KernelFormatterHandler
    """

    def __init__(self, accumulator_node_name):
        self.accumulator_node_name = accumulator_node_name
        self.output = IndentedBuffer(0)
        self.var_counter = 0
        self.aliases = dict()

    @staticmethod
    def ir_to_evt_argument_string(
        template_output_node_name: str,
        epilogue_nodes: List[IRNode],
    ):
        formatter = CutlassEVTEpilogueArgumentFormatter(
            template_output_node_name,
        )

        with virtualized.V.set_ops_handler(formatter), patch.object(  # type: ignore[call-arg]
            FlexibleLayout, "allow_indexing", True
        ):
            for node in epilogue_nodes:
                if isinstance(node, ComputedBuffer):
                    pnode = node.data
                else:
                    raise RuntimeError(
                        "Epilogue nodes must be Pointwise nodes, wrapped in a named ComputedBuffer"
                    )
                assert isinstance(pnode, Pointwise)
                pnode = cast(Pointwise, pnode)  # make mypy happy
                index = pnode._index(pnode.ranges)
                result = pnode.inner_fn(index)
                # each epilogue node results in a single "using" statement and may refer to the previous steps by name
                formatter.aliases[node.name] = result
            return formatter.getvalue(result)

    def __getattr__(self, name):
        def inner(*args, **kwargs):
            fargs = [_arg_str(a) for a in args]
            fkwargs = {key: _arg_str(a) for key, a in kwargs.items()}
            fn = getattr(self, f"_op_{name}")
            line = fn(*fargs, **fkwargs)
            return line

        if hasattr(self, f"_op_{name}"):
            return inner
        else:
            raise NotImplementedError(name)

    def _op_load(self, name, index_expr):
        if name == self.accumulator_node_name:
            return "{}"
        elif name in self.aliases:
            return self.aliases[name]
        else:
            return f"{name}"

    def _op_constant(self, value, dtype):
        if str(dtype) in ("torch.float16", "torch.float32"):
            return "{ static_cast<ElementAcc>(" + str(value) + ") }"
        else:
            raise NotImplementedError(f"Unsupported dtype for constant: {dtype}")

    def _cutlass_binary_functional_op(self, op, a, b):
        return "{" + str(a) + ", " + str(b) + "}"

    def _op_mul(self, a, b):
        return self._cutlass_binary_functional_op("multiplies", a, b)

    def _op_ge(self, a, b):
        return self._cutlass_binary_functional_op("greater_equal", a, b)

    def _op_add(self, a, b):
        return self._cutlass_binary_functional_op("plus", a, b)

    def _op_sub(self, a, b):
        return self._cutlass_binary_functional_op("minus", a, b)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise NotImplementedError()

    def getvalue(self, result):
        return result


#
# Copied and modified from https://github.com/NVIDIA/cutlass/blob/main/tools/library/scripts/gemm_operation.py
