from collections.abc import Callable
from typing import Any, Optional, Union

from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj, OptTensor


class AdvancedGNN(nn.Module):
    """Customized GNN based on torch_geometric BasicGNN

    It adds the possibility:
    - to repeat the same layer instead of duplicate it (only 1 set of parameters to learn)
    - to start with a (linear) encoding layer
    - to end with  a (linear) decoding layer

    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        message_passing_cls: type[MessagePassing],
        supports_edge_weight: bool = False,
        supports_edge_attr: bool = False,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        repeat_hidden_layer: bool = True,
        using_encoder: bool = True,
        using_decoder: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """

        Args:
            in_channels: input dim
            hidden_channels: dim of hidden layers
            num_layers: number of hidden layers
            message_passing_cls: MessagePassing subclass used to init each layer
            supports_edge_weight: supports message passing with one-dimensional edge weight
            supports_edge_attr: supports message passing with multi-dimensional edge feature
            out_channels: dim of output layer. If None, no output layer is added
            dropout: dropout probability (use after each layer, except for the output layer)
            repeat_hidden_layer: whether to pass repeatedly (if True) through the same hidden layer
                or to create several layers
            using_encoder: if True the first input layer is a classical MLP on node features,
                else this is another message passing layer
            using_decoder: if True and out_channels is not None, the output layer is a classical MLP on node features,
                else this is another message passing layer (or None if out_channels is None)
            act: The non-linear activation function to use after each layer, except for the output layer
                (existing only if out_channels is not None).
            act_kwargs: Arguments passed to the respective activation function defined by `act`.
            **kwargs: passed to message_passing_cls constructor.

        """
        super().__init__()

        self.supports_edge_attr = supports_edge_attr
        self.supports_edge_weight = supports_edge_weight
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.using_encoder = using_encoder
        self.using_decoder = using_decoder
        self.message_passing_cls = message_passing_cls
        self.repeat_hidden_layer = repeat_hidden_layer

        self.dropout = nn.Dropout(p=dropout)
        self.act = activation_resolver(act, **(act_kwargs or {}))

        # all convolutional for forward pass
        self._convs_forward = []

        # encoding first layer
        if self.using_encoder:
            self.first_layer = nn.Linear(
                in_features=in_channels, out_features=hidden_channels
            )
        else:
            self.first_layer = self.init_conv(
                in_channels=in_channels, out_channels=hidden_channels, **kwargs
            )
            self._convs_forward.append(self.first_layer)

        # hidden layer(s)
        if repeat_hidden_layer:
            # store convolution layers in an attribute to make its parameters be followed
            self.conv = self.init_conv(
                in_channels=hidden_channels, out_channels=hidden_channels, **kwargs
            )
            self._convs_forward += [self.conv] * self.num_layers  # repeat the layer
            self.convs = []
        else:
            self.conv = None
            # store convolution layers in module list to make their parameters be followed
            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                conv = self.init_conv(
                    in_channels=hidden_channels, out_channels=hidden_channels, **kwargs
                )
                self.convs.append(conv)
                self._convs_forward.append(conv)

        # last layer (if out_channels is not None)
        if self.out_channels is None:
            self.last_layer = None
        else:
            if self.using_decoder:
                self.last_layer = nn.Linear(
                    in_features=hidden_channels, out_features=out_channels
                )
            else:
                self.last_layer = self.init_conv(
                    in_channels=hidden_channels, out_channels=out_channels, **kwargs
                )
                self._convs_forward.append(self.last_layer)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
    ) -> Tensor:
        """Forward pass

        Args:
            x: input node features
            edge_index: edge indices
            edge_weight: edge weights
            edge_attr:  edge featrues

        Returns:
            node embeddings

        """
        if self.using_encoder:
            x = self.first_layer(x)
            x = self.act(x)
            x = self.dropout(x)

        for i_conv, conv in enumerate(self._convs_forward):
            if self.supports_edge_weight and self.supports_edge_attr:
                x = conv(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = conv(x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            if (
                i_conv < len(self._convs_forward) - 1
                or self.last_layer is None
                or self.using_decoder
            ):
                x = self.act(x)
                x = self.dropout(x)

        if self.using_decoder and self.last_layer is not None:
            x = self.last_layer(x)

        return x

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        if self.conv is not None:
            self.conv.reset_parameters()
        self.first_layer.reset_parameters()
        if self.last_layer is not None:
            self.last_layer.reset_parameters()

    def init_conv(
        self, in_channels: Union[int, tuple[int, int]], out_channels: int, **kwargs
    ) -> MessagePassing:
        return self.message_passing_cls(in_channels, out_channels, **kwargs)
