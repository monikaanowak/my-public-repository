from typing import Dict, Tuple, Type, Optional

from torch import Tensor, cat, mean, float32
from torch.nn import Sequential, Module, SELU, Conv2d, BatchNorm1d, BatchNorm2d, AvgPool2d, MaxPool2d, AlphaDropout, Linear



def reduce_tensor_dims(x: Tensor, dims: Dict[int, int], reduction=mean):
    """
    Reduce tensor dimensions by applying reduction function to them
    """
    reduce_dims = []
    view_dims = []

    shape_len = len(x.shape)
    for dim, size in enumerate(x.shape):
        dim_neg = dim - shape_len
        new_size = None
        if dim in dims:
            new_size = dims[dim]
        elif dim_neg in dims:
            new_size = dims[dim_neg]

        if new_size is None or size == new_size:
            view_dims.append(size)
        else:
            view_dims.append(new_size)
            view_dims.append(size // new_size)
            reduce_dims.append(len(view_dims) - 1)

    if reduce_dims:
        return reduction(x.view(*view_dims), dim=tuple(reduce_dims))
    else:
        return x


def expand_tensor_to(x: Tensor, dims: Dict[int, int]):
    """
    Expand tensor dimensions by repeating them to new size (new size must be divisible by old size)
    """
    view_dims = []
    expand_dims = []
    to_dims = []

    shape_len = len(x.shape)
    for dim, size in enumerate(x.shape):
        dim_neg = dim - shape_len
        new_size = None
        if dim in dims:
            new_size = dims[dim]
        elif dim_neg in dims:
            new_size = dims[dim_neg]

        if new_size is None or size == new_size:
            view_dims.append(size)
            expand_dims.append(size)
            to_dims.append(size)
        else:
            if new_size % size != 0:
                raise ValueError(f'Cannot expand {x.shape} => {dims} - invalid dims {size} => {new_size}')

            view_dims.append(size)
            view_dims.append(1)
            expand_dims.append(size)
            expand_dims.append(new_size // size)
            to_dims.append(new_size)

    return x.view(*view_dims).expand(*expand_dims).reshape(*to_dims)


def conform_tensor_to(x: Tensor, to_shape: Tuple[int, ...]):
    """
    Conform tensor to new shape by reducing or expanding dimensions
    """
    reduces = {}
    expands = {}

    for dim, (x_size, to_size) in enumerate(zip(x.shape, to_shape)):
        if x_size < to_size:
            expands[dim] = to_size
        elif x_size > to_size:
            reduces[dim] = to_size

    if reduces:
        x = reduce_tensor_dims(x, reduces)
    if expands:
        x = expand_tensor_to(x, expands)

    return x


class ConvReduction(Sequential):
    """
    Perform some operations on reduced size image and then expand features back to original size
    """

    def forward(self, input: Tensor) -> Tensor:
        block_out = super().forward(input)
        block_out = conform_tensor_to(block_out, input.shape)
        return input + block_out


class ResidualSequential(Sequential):
    """
    Run blocks sequentially and pass input to output of each block
    """

    def forward(self, input: Tensor) -> Tensor:
        for block in self:
            block_out = block.forward(input)
            input = conform_tensor_to(input, block_out.shape) + block_out
        return input


class Parallel(Sequential):
    """
    Run blocks in parallel and sum their outputs
    """

    def forward(self, input: Tensor) -> Tensor:
        return sum(block.forward(input) for block in self)


class ConvBlock(Module):
    """
    Convolutional block with normalization and activation - typical convolutional layer usage
    """

    def __init__(
            self, in_features, out_features,
            kernel_size=(1, 1),
            activation_cls: Optional[Type[Module]] = SELU,
            normalize=True,
    ):
        super().__init__()
        layers = []
        layers.append(Conv2d(
            in_features, out_features,
            kernel_size=kernel_size, padding='same',
        ))
        if normalize:
            layers.append(BatchNorm2d(out_features))
        if activation_cls is not None:
            layers.append(activation_cls(inplace=True))
        self.net = Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.net.forward(input)


class LinearBlock(Module):
    """
    Linear block with normalization and activation - typical linear layer usage
    """

    def __init__(
            self, in_features, out_features,
            activation_cls: Optional[Type[Module]] = SELU,
            normalize=True,
    ):
        super().__init__()
        layers = []
        layers.append(Linear(in_features, out_features))
        if normalize:
            layers.append(BatchNorm1d(out_features))
        if activation_cls is not None:
            layers.append(activation_cls(inplace=True))
        self.net = Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.net.forward(input)


class ParallelConvStrip(Module):
    """
    Perform convolution on strip shape kernel in two directions and sum results
    """

    def __init__(
            self, in_features, out_features, strip_size,
            activation_cls: Optional[Type[Module]] = SELU,
            normalize=True,
    ):
        super().__init__()
        self.net = Parallel(
            ConvBlock(
                in_features, out_features, kernel_size=(strip_size, 1),
                normalize=normalize, activation_cls=activation_cls
            ),
            ConvBlock(
                in_features, out_features, kernel_size=(1, strip_size),
                normalize=normalize, activation_cls=activation_cls
            ),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.net.forward(input)


class CornerModel(Module):
    def __init__(self):
        n = 64
        super().__init__()
        self.net = Sequential(
            AlphaDropout(0.1),
            ConvBlock(3, 16, kernel_size=(3, 3)),
            ConvReduction(
                AvgPool2d(3),
                ParallelConvStrip(16, 16, 3),
                AlphaDropout(0.1),
                ParallelConvStrip(16, 32, 3),
                ConvReduction(
                    MaxPool2d(3),
                    ResidualSequential(
                        ParallelConvStrip(32, 32, 3),
                        ParallelConvStrip(32, 32, 3),
                        ParallelConvStrip(32, 32, 3),
                    ),
                    AlphaDropout(0.1),
                    ParallelConvStrip(32, n, 3),
                    ConvReduction(
                        AvgPool2d(3),
                        ResidualSequential(
                            ParallelConvStrip(n, n, 5),
                            ParallelConvStrip(n, n, 5),
                            ParallelConvStrip(n, n, 5),
                            ParallelConvStrip(n, n, 3),
                            ParallelConvStrip(n, n, 3),
                            ParallelConvStrip(n, n, 3),
                            ParallelConvStrip(n, n, 3),
                            ParallelConvStrip(n, n, 3),
                        ),
                        ParallelConvStrip(n, 32, 3),
                    ),
                ),
                ResidualSequential(
                    ParallelConvStrip(32, 32, 3),
                    ParallelConvStrip(32, 32, 3),
                ),
                ParallelConvStrip(32, 16, 3),
            ),
            ResidualSequential(
                ParallelConvStrip(16, 16, 3),
            ),
            ParallelConvStrip(16, 1, 3, activation_cls=None),
        )

    def forward(self, input: Tensor) -> Tensor:
        input = input.permute(0, 3, 1, 2).to(dtype=float32)
        return self.net.forward(input).abs()


LETTERS = tuple('-~aąbcćdeęfghijklłmnńoóprsśtuwyzźż')
LETTER_SHAPE = (108, 108)


class LetterClassifier(Module):
    def __init__(self):
        super().__init__()
        n = 64 + 32
        self.net = Sequential(
            ParallelConvStrip(3, 32, 3),
            ResidualSequential(
                ParallelConvStrip(32, 32, 1),
            ),
            AlphaDropout(0.15),
            MaxPool2d(3),
            ResidualSequential(
                ParallelConvStrip(32, 32, 3),
                ParallelConvStrip(32, 32, 3),
                ParallelConvStrip(32, 32, 3),
            ),
            AvgPool2d(2),
            ResidualSequential(
                ParallelConvStrip(32, 32, 3),
                ParallelConvStrip(32, 32, 3),
                ParallelConvStrip(32, 32, 3),
                ParallelConvStrip(32, 64, 3),
            ),
            AlphaDropout(0.15),
            MaxPool2d(2),
            ParallelConvStrip(64, n, 3),
            ResidualSequential(
                ParallelConvStrip(n, n, 5),
                ParallelConvStrip(n, n, 3),
                ParallelConvStrip(n, n, 5),
                ParallelConvStrip(n, n, 3),
            ),
        )
        self.net_linear = Sequential(
            AlphaDropout(.05),
            ResidualSequential(
                LinearBlock(n * 3, n * 3),
                LinearBlock(n * 3, n),
                LinearBlock(n, n),
                LinearBlock(n, n),
            ),
            Linear(n, len(LETTERS)),
        )

    def forward(self, input: Tensor) -> Tensor:
        input = input.permute(0, 3, 1, 2).to(dtype=float32)
        conv_out = self.net.forward(input)

        dims_merged = conv_out.reshape(*conv_out.shape[:2], -1)
        reduced = cat((
            dims_merged.mean(dim=-1),
            dims_merged.amin(dim=-1), dims_merged.amax(dim=-1)
        ), dim=-1)

        return self.net_linear.forward(reduced)
