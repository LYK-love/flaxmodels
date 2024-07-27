import jax
import jax.numpy as jnp
import flax.linen as nn
import functools
from typing import Any, Callable, Iterable, Optional, Tuple, Union, List
import h5py
import warnings
from . import ops
from .. import utils
from dataclasses import dataclass, field


URLS = {
    "resnet18": "https://www.dropbox.com/s/wx3vt76s5gpdcw5/resnet18_weights.h5?dl=1",
    "resnet34": "https://www.dropbox.com/s/rnqn2x6trnztg4c/resnet34_weights.h5?dl=1",
    "resnet50": "https://www.dropbox.com/s/fcc8iii38ezvqog/resnet50_weights.h5?dl=1",
    "resnet101": "https://www.dropbox.com/s/hgtnk586pnz0xug/resnet101_weights.h5?dl=1",
    "resnet152": "https://www.dropbox.com/s/tvi28uwiy54mcfr/resnet152_weights.h5?dl=1",
}

LAYERS = {
    "resnet18": [2, 2, 2, 2],
    "resnet34": [3, 4, 6, 3],
    "resnet50": [3, 4, 6, 3],
    "resnet101": [3, 4, 23, 3],
    "resnet152": [3, 8, 36, 3],
}


def observe(module: nn.Module, x: jax.Array, idx: int = 4) -> jax.Array:
    """
    A flax module layer for Grad-CAM observation of the output that is passed in.
    Returns x with no changes, this simply adds model parameters for storing values for the Grad-CAM.

    Arguments:
    - module: Flax module to make the observation of
    - x: forward pass value to observe
    """

    x = module.perturb(f"gradcam_perturb_{idx}", x)
    module.sow("intermediates", f"gradcam_sow_{idx}", x)
    return x


class BasicBlock(nn.Module):
    """
    Basic Block.

    Attributes:
        features (int): Number of output channels.
        kernel_size (Tuple): Kernel size.
        downsample (bool): If True, downsample spatial resolution.
        stride (bool): If True, use strides (2, 2). Not used in this module.
                       The attribute is only here for compatibility with Bottleneck.
        param_dict (h5py.Group): Parameter dict with pretrained parameters.
        kernel_init (functools.partial): Kernel initializer.
        bias_init (functools.partial): Bias initializer.
        block_name (str): Name of block.
        dtype (str): Data type.
    """

    features: int
    customized_relu: Callable = nn.relu
    kernel_size: Union[int, Iterable[int]] = (3, 3)
    downsample: bool = False
    stride: bool = True
    param_dict: h5py.Group = None
    kernel_init: functools.partial = nn.initializers.lecun_normal()
    bias_init: functools.partial = nn.initializers.zeros
    block_name: str = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, act, train=True, observe_layers_in_block: List[int] = []):
        """
        Run Basic Block.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].
            act (dict): Dictionary containing activations.
            train (bool): Training mode.

        Returns:
            (tensor): Output shape of shape [N, H', W', features].
        """
        residual = x

        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=(2, 2) if self.downsample else (1, 1),
            padding=((1, 1), (1, 1)),
            kernel_init=(
                self.kernel_init
                if self.param_dict is None
                else lambda *_: jnp.array(self.param_dict["conv1"]["weight"])
            ),
            use_bias=False,
            dtype=self.dtype,
        )(x)

        if len(observe_layers_in_block) > 0 and 1 in observe_layers_in_block:
            x = observe(self, x, idx=1)

        x = ops.batch_norm(
            x,
            train=train,
            epsilon=1e-05,
            momentum=0.1,
            params=None if self.param_dict is None else self.param_dict["bn1"],
            dtype=self.dtype,
        )
        x = self.customized_relu(x)

        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            kernel_init=(
                self.kernel_init
                if self.param_dict is None
                else lambda *_: jnp.array(self.param_dict["conv2"]["weight"])
            ),
            use_bias=False,
            dtype=self.dtype,
        )(x)

        if len(observe_layers_in_block) > 0 and 2 in observe_layers_in_block:
            x = observe(self, x, idx=2)

        x = ops.batch_norm(
            x,
            train=train,
            epsilon=1e-05,
            momentum=0.1,
            params=None if self.param_dict is None else self.param_dict["bn2"],
            dtype=self.dtype,
        )

        if self.downsample:
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(2, 2),
                kernel_init=(
                    self.kernel_init
                    if self.param_dict is None
                    else lambda *_: jnp.array(
                        self.param_dict["downsample"]["conv"]["weight"]
                    )
                ),
                use_bias=False,
                dtype=self.dtype,
            )(residual)

            residual = ops.batch_norm(
                residual,
                train=train,
                epsilon=1e-05,
                momentum=0.1,
                params=(
                    None
                    if self.param_dict is None
                    else self.param_dict["downsample"]["bn"]
                ),
                dtype=self.dtype,
            )

        x += residual
        x = self.customized_relu(x)
        act[self.block_name] = x
        return x


class Bottleneck(nn.Module):
    """
    Bottleneck.

    Attributes:
        features (int): Number of output channels.
        kernel_size (Tuple): Kernel size.
        downsample (bool): If True, downsample spatial resolution.
        stride (bool): If True, use strides (2, 2). Not used in this module.
                       The attribute is only here for compatibility with Bottleneck.
        param_dict (h5py.Group): Parameter dict with pretrained parameters.
        kernel_init (functools.partial): Kernel initializer.
        bias_init (functools.partial): Bias initializer.
        block_name (str): Name of block.
        expansion (int): Factor to multiply number of output channels with.
        dtype (str): Data type.
    """

    features: int
    customized_relu: Callable = nn.relu
    kernel_size: Union[int, Iterable[int]] = (3, 3)
    downsample: bool = False
    stride: bool = True
    param_dict: Any = None
    kernel_init: functools.partial = nn.initializers.lecun_normal()
    bias_init: functools.partial = nn.initializers.zeros
    block_name: str = None
    expansion: int = 4
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, act, train=True, observe_layers_in_block: List[int] = []):
        """
        Run Bottleneck.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].
            act (dict): Dictionary containing activations.
            train (bool): Training mode.

        Returns:
            (tensor): Output shape of shape [N, H', W', features].
        """
        residual = x

        x = nn.Conv(
            features=self.features,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_init=(
                self.kernel_init
                if self.param_dict is None
                else lambda *_: jnp.array(self.param_dict["conv1"]["weight"])
            ),
            use_bias=False,
            dtype=self.dtype,
        )(x)

        if len(observe_layers_in_block) > 0 and 1 in observe_layers_in_block:
            x = observe(self, x, idx=1)

        x = ops.batch_norm(
            x,
            train=train,
            epsilon=1e-05,
            momentum=0.1,
            params=None if self.param_dict is None else self.param_dict["bn1"],
            dtype=self.dtype,
        )
        x = self.customized_relu(x)

        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(2, 2) if self.downsample and self.stride else (1, 1),
            padding=((1, 1), (1, 1)),
            kernel_init=(
                self.kernel_init
                if self.param_dict is None
                else lambda *_: jnp.array(self.param_dict["conv2"]["weight"])
            ),
            use_bias=False,
            dtype=self.dtype,
        )(x)

        if len(observe_layers_in_block) > 0 and 2 in observe_layers_in_block:
            x = observe(self, x, idx=2)

        x = ops.batch_norm(
            x,
            train=train,
            epsilon=1e-05,
            momentum=0.1,
            params=None if self.param_dict is None else self.param_dict["bn2"],
            dtype=self.dtype,
        )
        x = self.customized_relu(x)

        x = nn.Conv(
            features=self.features * self.expansion,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_init=(
                self.kernel_init
                if self.param_dict is None
                else lambda *_: jnp.array(self.param_dict["conv3"]["weight"])
            ),
            use_bias=False,
            dtype=self.dtype,
        )(x)

        if len(observe_layers_in_block) > 0 and 3 in observe_layers_in_block:
            x = observe(self, x, idx=3)

        x = ops.batch_norm(
            x,
            train=train,
            epsilon=1e-05,
            momentum=0.1,
            params=None if self.param_dict is None else self.param_dict["bn3"],
            dtype=self.dtype,
        )

        if self.downsample:
            residual = nn.Conv(
                features=self.features * self.expansion,
                kernel_size=(1, 1),
                strides=(2, 2) if self.stride else (1, 1),
                kernel_init=(
                    self.kernel_init
                    if self.param_dict is None
                    else lambda *_: jnp.array(
                        self.param_dict["downsample"]["conv"]["weight"]
                    )
                ),
                use_bias=False,
                dtype=self.dtype,
            )(residual)

            residual = ops.batch_norm(
                residual,
                train=train,
                epsilon=1e-05,
                momentum=0.1,
                params=(
                    None
                    if self.param_dict is None
                    else self.param_dict["downsample"]["bn"]
                ),
                dtype=self.dtype,
            )

        x += residual
        x = self.customized_relu(x)
        act[self.block_name] = x
        return x


class ResNet(nn.Module):
    """
    ResNet.

    Attributes:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000]
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000]
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        architecture (str):
            Which ResNet model to use:
                - 'resnet18'
                - 'resnet34'
                - 'resnet50'
                - 'resnet101'
                - 'resnet152'
        num_classes (int):
            Number of classes.
        block (nn.Module):
            Type of residual block:
                - BasicBlock
                - Bottleneck
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used.
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.
    """

    output: str = "softmax"
    pretrained: str = "imagenet"
    normalize: bool = True
    architecture: str = "resnet18"
    num_classes: int = 1000
    block: nn.Module = BasicBlock
    kernel_init: functools.partial = nn.initializers.lecun_normal()
    bias_init: functools.partial = nn.initializers.zeros
    ckpt_dir: str = None
    dtype: str = "float32"

    customized_relu: Callable = nn.relu
    observe_layers: List[int] = field(
        default_factory=lambda: [
            4,
        ]
    )
    observe_layers_in_last_block: List[int] = field(default_factory=lambda: [])

    def setup(self):
        self.param_dict = None
        if self.pretrained == "imagenet":
            ckpt_file = utils.download(self.ckpt_dir, URLS[self.architecture])
            self.param_dict = h5py.File(ckpt_file, "r")

    @nn.compact
    def __call__(self, x, train=True):
        """
        Args:
            x (tensor): Input tensor of shape [N, H, W, 3]. Images must be in range [0, 1].
            train (bool): Training mode.

        Returns:
            (tensor): Out
            If output == 'logits' or output == 'softmax':
                (tensor): Output tensor of shape [N, num_classes].
            If output == 'activations':
                (dict): Dictionary of activations.
        """
        if self.normalize:
            mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, -1).astype(x.dtype)
            std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, -1).astype(x.dtype)
            x = (x - mean) / std

        if self.pretrained == "imagenet":
            if self.num_classes != 1000:
                warnings.warn(
                    f"The user specified parameter 'num_classes' was set to {self.num_classes} "
                    "but will be overwritten with 1000 to match the specified pretrained checkpoint 'imagenet', if ",
                    UserWarning,
                )
            num_classes = 1000
        else:
            num_classes = self.num_classes

        act = {}

        x = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            kernel_init=(
                self.kernel_init
                if self.param_dict is None
                else lambda *_: jnp.array(self.param_dict["conv1"]["weight"])
            ),
            strides=(2, 2),
            padding=((3, 3), (3, 3)),
            use_bias=False,
            dtype=self.dtype,
        )(x)
        act["conv1"] = x

        x = ops.batch_norm(
            x,
            train=train,
            epsilon=1e-05,
            momentum=0.1,
            params=None if self.param_dict is None else self.param_dict["bn1"],
            dtype=self.dtype,
        )
        x = self.customized_relu(x)
        x = nn.max_pool(
            x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
        )

        # Layer 1
        down = self.block.__name__ == "Bottleneck"
        for i in range(LAYERS[self.architecture][0]):
            params = (
                None
                if self.param_dict is None
                else self.param_dict["layer1"][f"block{i}"]
            )
            x = self.block(
                features=64,
                customized_relu=self.customized_relu,
                kernel_size=(3, 3),
                downsample=i == 0 and down,
                stride=i != 0,
                param_dict=params,
                block_name=f"block1_{i}",
                dtype=self.dtype,
            )(x, act, train)

        if len(self.observe_layers) > 0 and 1 in self.observe_layers:
            x = observe(self, x, idx=1)

        # Layer 2
        for i in range(LAYERS[self.architecture][1]):
            params = (
                None
                if self.param_dict is None
                else self.param_dict["layer2"][f"block{i}"]
            )
            x = self.block(
                features=128,
                customized_relu=self.customized_relu,
                kernel_size=(3, 3),
                downsample=i == 0,
                param_dict=params,
                block_name=f"block2_{i}",
                dtype=self.dtype,
            )(x, act, train)

        if len(self.observe_layers) > 0 and 2 in self.observe_layers:
            x = observe(self, x, idx=2)

        # Layer 3
        for i in range(LAYERS[self.architecture][2]):
            params = (
                None
                if self.param_dict is None
                else self.param_dict["layer3"][f"block{i}"]
            )
            x = self.block(
                features=256,
                customized_relu=self.customized_relu,
                kernel_size=(3, 3),
                downsample=i == 0,
                param_dict=params,
                block_name=f"block3_{i}",
                dtype=self.dtype,
            )(x, act, train)

        if len(self.observe_layers) > 0 and 3 in self.observe_layers:
            x = observe(self, x, idx=3)

        # Layer 4
        # We also insert observe layer into the last block,
        for i in range(LAYERS[self.architecture][3]):
            params = (
                None
                if self.param_dict is None
                else self.param_dict["layer4"][f"block{i}"]
            )

            if i == LAYERS[self.architecture][3] - 1 and (
                len(self.observe_layers_in_last_block) != 0
            ):
                x = self.block(
                    features=512,
                    customized_relu=self.customized_relu,
                    kernel_size=(3, 3),
                    downsample=i == 0,
                    param_dict=params,
                    block_name=f"block4_{i}",
                    dtype=self.dtype,
                )(
                    x,
                    act,
                    train,
                    observe_layers_in_block=self.observe_layers_in_last_block,
                )
            else:
                x = self.block(
                    features=512,
                    customized_relu=self.customized_relu,
                    kernel_size=(3, 3),
                    downsample=i == 0,
                    param_dict=params,
                    block_name=f"block4_{i}",
                    dtype=self.dtype,
                )(x, act, train)

        x = x  # For debugging purposes

        # Add observed layer for Grad-CAM
        if len(self.observe_layers) > 0 and 4 in self.observe_layers:
            x = observe(self, x, idx=4)

        # Classifier
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(
            features=num_classes,
            kernel_init=(
                self.kernel_init
                if self.param_dict is None
                else lambda *_: jnp.array(self.param_dict["fc"]["weight"])
            ),
            bias_init=(
                self.bias_init
                if self.param_dict is None
                else lambda *_: jnp.array(self.param_dict["fc"]["bias"])
            ),
            dtype=self.dtype,
        )(x)
        act["fc"] = x

        if self.output == "softmax":
            return nn.softmax(x)
        if self.output == "log_softmax":
            return nn.log_softmax(x)
        if self.output == "activations":
            return act
        return x


def ResNet18(
    output="softmax",
    pretrained="imagenet",
    normalize=True,
    num_classes=1000,
    kernel_init=nn.initializers.lecun_normal(),
    bias_init=nn.initializers.zeros,
    ckpt_dir=None,
    dtype="float32",
    customized_relu: Callable = nn.relu,
    observe_layers: List[int] = field(default_factory=lambda: [4]),
    observe_layers_in_last_block: List[int] = field(default_factory=lambda: []),
):
    """
    Implementation of the ResNet18 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000]
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000]
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used.
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(
        output=output,
        pretrained=pretrained,
        normalize=normalize,
        architecture="resnet18",
        num_classes=num_classes,
        block=BasicBlock,
        kernel_init=kernel_init,
        bias_init=bias_init,
        ckpt_dir=ckpt_dir,
        dtype=dtype,
        customized_relu=customized_relu,
        observe_layers=observe_layers,
        observe_layers_in_last_block=observe_layers_in_last_block,
    )


def ResNet34(
    output="softmax",
    pretrained="imagenet",
    normalize=True,
    num_classes=1000,
    kernel_init=nn.initializers.lecun_normal(),
    bias_init=nn.initializers.zeros,
    ckpt_dir=None,
    dtype="float32",
    customized_relu: Callable = nn.relu,
    observe_layers: List[int] = field(default_factory=lambda: [4]),
    observe_layers_in_last_block: List[int] = field(default_factory=lambda: []),
):
    """
    Implementation of the ResNet34 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000]
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000]
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used.
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(
        output=output,
        pretrained=pretrained,
        normalize=normalize,
        architecture="resnet34",
        num_classes=num_classes,
        block=BasicBlock,
        kernel_init=kernel_init,
        bias_init=bias_init,
        ckpt_dir=ckpt_dir,
        dtype=dtype,
        customized_relu=customized_relu,
        observe_layers=observe_layers,
        observe_layers_in_last_block=observe_layers_in_last_block,
    )


def ResNet50(
    output="softmax",
    pretrained="imagenet",
    normalize=True,
    num_classes=1000,
    kernel_init=nn.initializers.lecun_normal(),
    bias_init=nn.initializers.zeros,
    ckpt_dir=None,
    dtype="float32",
    customized_relu: Callable = nn.relu,
    observe_layers: List[int] = field(default_factory=lambda: [4]),
    observe_layers_in_last_block: List[int] = field(default_factory=lambda: []),
):
    """
    Implementation of the ResNet50 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000]
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000]
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used.
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(
        output=output,
        pretrained=pretrained,
        normalize=normalize,
        architecture="resnet50",
        num_classes=num_classes,
        block=Bottleneck,
        kernel_init=kernel_init,
        bias_init=bias_init,
        ckpt_dir=ckpt_dir,
        dtype=dtype,
        customized_relu=customized_relu,
        observe_layers=observe_layers,
        observe_layers_in_last_block=observe_layers_in_last_block,
    )


def ResNet101(
    output="softmax",
    pretrained="imagenet",
    normalize=True,
    num_classes=1000,
    kernel_init=nn.initializers.lecun_normal(),
    bias_init=nn.initializers.zeros,
    ckpt_dir=None,
    dtype="float32",
    customized_relu: Callable = nn.relu,
    observe_layers: List[int] = field(default_factory=lambda: [4]),
    observe_layers_in_last_block: List[int] = field(default_factory=lambda: []),
):
    """
    Implementation of the ResNet101 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000]
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000]
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used.
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(
        output=output,
        pretrained=pretrained,
        normalize=normalize,
        architecture="resnet101",
        num_classes=num_classes,
        block=Bottleneck,
        kernel_init=kernel_init,
        bias_init=bias_init,
        ckpt_dir=ckpt_dir,
        dtype=dtype,
        customized_relu=customized_relu,
        observe_layers=observe_layers,
        observe_layers_in_last_block=observe_layers_in_last_block,
    )


def ResNet152(
    output="softmax",
    pretrained="imagenet",
    normalize=True,
    num_classes=1000,
    kernel_init=nn.initializers.lecun_normal(),
    bias_init=nn.initializers.zeros,
    ckpt_dir=None,
    dtype="float32",
    customized_relu: Callable = nn.relu,
    observe_layers: List[int] = field(default_factory=lambda: [4]),
    observe_layers_in_last_block: List[int] = field(default_factory=lambda: []),
):
    """
    Implementation of the ResNet152 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000]
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000]
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used.
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(
        output=output,
        pretrained=pretrained,
        normalize=normalize,
        architecture="resnet152",
        num_classes=num_classes,
        block=Bottleneck,
        kernel_init=kernel_init,
        bias_init=bias_init,
        ckpt_dir=ckpt_dir,
        dtype=dtype,
        customized_relu=customized_relu,
        observe_layers=observe_layers,
        observe_layers_in_last_block=observe_layers_in_last_block,
    )
