import torch as th
import torch_geometric as thg
from pytest_cases import fixture, param_fixture

from skdecide.hub.solver.utils.gnn.advanced_gnn import AdvancedGNN


@fixture
def data():
    edge_index = th.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=th.long)
    x = th.tensor([[-1, 1], [0, 1], [1, 5]], dtype=th.float)
    edge_weight = th.tensor([0, 1, 2, 3], dtype=th.float)
    edge_attr = th.ones((4, 2))
    return thg.data.Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight
    )


out_channels = param_fixture("out_channels", [4, None])
using_encoder = param_fixture("using_encoder", [True, False])
using_decoder = param_fixture("using_decoder", [True, False])
repeat_hidden_layer = param_fixture("repeat_hidden_layer", [True, False])


def test_advanced_gnn(
    data, out_channels, using_encoder, using_decoder, repeat_hidden_layer
):
    x, edge_index, edge_attr, edge_weight = (
        data.x,
        data.edge_index,
        data.edge_attr,
        data.edge_weight,
    )
    in_channels = x.shape[1]
    hidden_channels = 16
    num_layers = 3
    message_passing_cls = thg.nn.GCNConv
    supports_edge_weight = True
    supports_edge_attr = False

    gnn = AdvancedGNN(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        message_passing_cls=message_passing_cls,
        supports_edge_weight=supports_edge_weight,
        supports_edge_attr=supports_edge_attr,
        using_encoder=using_encoder,
        using_decoder=using_decoder,
        repeat_hidden_layer=repeat_hidden_layer,
    )

    if using_encoder:
        assert isinstance(gnn.first_layer, th.nn.Linear)
    else:
        assert isinstance(gnn.first_layer, message_passing_cls)
    if out_channels is None:
        assert gnn.last_layer is None
    else:
        if using_decoder:
            assert isinstance(gnn.last_layer, th.nn.Linear)
        else:
            assert isinstance(gnn.last_layer, message_passing_cls)
    if repeat_hidden_layer:
        assert isinstance(gnn.conv, thg.nn.GCNConv)
        assert len(gnn.convs) == 0
    else:
        assert gnn.conv is None
        assert len(gnn.convs) == num_layers
        assert isinstance(gnn.convs[0], thg.nn.GCNConv)

    n_params = 2  # first layer
    if repeat_hidden_layer:
        n_params += 2  # hidden layer
    else:
        n_params += 2 * (num_layers)  # hidden layers
    if out_channels is not None:
        n_params += 2  # last layer
    assert len(list(gnn.parameters())) == n_params

    y = gnn(x=x, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr)

    if out_channels is None:
        assert y.shape == (x.shape[0], hidden_channels)
    else:
        assert y.shape == (x.shape[0], out_channels)
