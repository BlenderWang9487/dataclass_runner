import dataclasses
import inspect
from collections import defaultdict
from typing import Union


class BaseConfig:
    @classmethod
    def from_config(cls, args_dict: Union[dict, "BaseConfig"]):
        if isinstance(args_dict, BaseConfig):
            args_dict = args_dict.__dict__
        return cls(
            **{
                k: v
                for k, v in args_dict.items()
                if k in inspect.signature(cls).parameters
            }
        )


runner_dataclass = dataclasses.dataclass(kw_only=True)


def check_conflicting_params(cls=None, /, ignore_check_names: list[str] | None = None):
    def _check_impl(cls, ignore_check_names: list[str] | None = None):
        # check if any base class parameters are conflicting
        if ignore_check_names is None:
            ignore_check_names = []
        ct = defaultdict(set)
        for base_class in cls.__bases__:
            if issubclass(base_class, BaseConfig):
                for name in inspect.signature(base_class).parameters.keys():
                    ct[name].add(base_class.__name__)
        for name, classes in ct.items():
            if len(classes) > 1 and name not in ignore_check_names:
                raise ValueError(f"Conflicting parameter name: {name} in {classes}")
        return cls

    def wrap(cls):
        return _check_impl(cls, ignore_check_names=ignore_check_names)

    if cls is None:
        return wrap

    return wrap(cls)
