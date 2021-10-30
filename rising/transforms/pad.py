from typing import Sequence

from rising.transforms.abstract import BaseTransform, item_or_seq
from rising.transforms.functional.pad import pad as _pad
from rising.transforms.functional.pad import pad_parameters


class Pad(BaseTransform):
    def __init__(
        self,
        *,
        padding_size: item_or_seq[int],
        mode: str = "constant",
        pad_value: item_or_seq[float],
        keys: Sequence[str] = ("data",),
        grad: bool = False,
        **kwargs
    ):
        """
        padding_size should be the same for all the keys.
        pad_value can be different for different keys
        """
        super().__init__(_pad, keys=keys, grad=grad, property_names=(), **kwargs)
        self.padding_size = padding_size
        self.mode = mode
        self.pad_value = self._tuple_generator(pad_value)

    def forward(self, **data) -> dict:
        """
        Apply transformation

        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        kwargs = {}
        for k in self.property_names:
            kwargs[k] = getattr(self, k)

        kwargs.update(self.kwargs)

        for _key, _value in zip(self.keys, self.pad_value):
            cur_data = data[_key]
            input_shape = cur_data.shape[2:]
            pad_size = pad_parameters(input_shape, self.padding_size, ndim=cur_data.dim() - 2)
            data[_key] = self.augment_fn(data[_key], pad_size=pad_size, grid_pad=False, mode=self.mode, value=_value)
        return data
