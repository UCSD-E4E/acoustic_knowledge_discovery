from torch import nn
from ..dataset import ChunkDataset
from .feature_preprocessor import FeaturePreprocessor

class FeatureSequential(nn.Module):
    """Sequentially applies a series of feature preprocessing steps to a KnowledgeDataset"""

    def __init__(self, *steps: FeaturePreprocessor):
        super().__init__()
        # Register steps as submodules so they’re tracked like PyTorch modules
        for idx, step in enumerate(steps):
            self.add_module(str(idx), step)

    def forward(self, kd: ChunkDataset) -> ChunkDataset:
        for name, step in self._modules.items():
            if not isinstance(step, FeaturePreprocessor):
                raise TypeError(f"Step {name} is not inheriting from FeaturePreprocessor")
            kd = step(kd)  # each step is a FeaturePreprocessor. this invokes __call__ method
        return kd
