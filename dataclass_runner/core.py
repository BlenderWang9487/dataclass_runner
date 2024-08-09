import dataclasses
import inspect
from collections import defaultdict
from typing import Any, Callable, Union


class BaseConfig:
    @classmethod
    def from_config(cls, args_dict: Union[dict, "BaseConfig"]):
        if isinstance(args_dict, BaseConfig):
            args_dict = args_dict.__dict__
        params = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in args_dict.items() if k in params})


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


class DataclassRunner:
    def __init__(
        self, app, callback: Callable[[BaseConfig], Any], name: str = None
    ) -> None:
        assert hasattr(
            app, "command"
        ), "app must have command wrapper method to wrap the runner_type"
        self._app = app
        self._callback = callback
        if name is None:
            name = callback.__name__
        self._name = name

    def __call__(self, runner_type):
        assert dataclasses.is_dataclass(runner_type) and issubclass(
            runner_type, BaseConfig
        ), f"runner_type must be a dataclass and subclass of {BaseConfig}, got {runner_type}"

        @dataclasses.dataclass
        class BindType(runner_type):  # type: ignore
            def __post_init__(bind_self):
                parent = super()
                if hasattr(parent, "__post_init__"):
                    parent.__post_init__()
                self._callback(runner_type.from_config(bind_self))

        self._app.command(name=self._name)(BindType)
        return runner_type
