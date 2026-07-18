# Utility files

The training script `train_afgm.py` depends on HVE utility functions and the modified AFGM attention module.

Please place the following files in this folder:

- `tools.py`
- `data_loader.py`
- `attention.py`

These files should provide:

```python
from util.tools import load_resnet18, get_latent_output
from util.data_loader import get_Dataloader
from util.attention import attention
```

If your implementation modifies the original HVE files, please keep the original HVE license notice and clearly document the modifications.
