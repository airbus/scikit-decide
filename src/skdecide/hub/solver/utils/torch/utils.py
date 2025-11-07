import numpy as np
import torch.nn


def extract_module_parameters_values(m: torch.nn.Module) -> dict[str, np.ndarray]:
    return {
        name: np.array(param.data.cpu().numpy()) for name, param in m.named_parameters()
    }
