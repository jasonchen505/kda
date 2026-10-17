"""Constructive host-compute checks and real JIT dispatch witnesses; not a security sandbox."""
import ast
import collections
from contextlib import contextmanager


def validate_host(source):
    tree=ast.parse(source)
    blocked={'matmul','mm','bmm','softmax','sum','mean','square','float','bfloat16','to','contiguous'}
    def host(node):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and any(
                isinstance(d,ast.Attribute) and d.attr=='jit' for d in node.decorator_list):
            return
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.MatMult):
            raise ValueError(f'Host matrix multiply at line{node.lineno}; compute must execute in Triton')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in blocked:
            raise ValueError(f'Host tensor compute .{node.func.attr} at line{node.lineno}; only allocation/metadata outside JIT')
        for child in ast.iter_child_nodes(node):
            host(child)
    host(tree)


@contextmanager
def track_dispatches(module_name):
    from triton.runtime.jit import JITFunction
    original=JITFunction.run
    counts=collections.Counter()
    def tracked(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        if self.fn.__module__==module_name and not kwargs.get('warmup',False):
            counts[self.fn.__name__]+=1
        return result
    JITFunction.run=tracked
    try:
        yield counts
    finally:
        JITFunction.run=original
