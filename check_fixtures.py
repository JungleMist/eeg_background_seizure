import sys
sys.path.insert(0, '.')
import numpy as np
from tests.conftest import CH_NAMES_19, BASE_CFG

# Test synthetic_epoch shape
rng = np.random.default_rng(42)
n_ch = len(CH_NAMES_19)
n_times = 1000
source = rng.standard_normal(n_times) * 50.0
gains = rng.uniform(0.5, 1.0, n_ch)
noise = rng.standard_normal((n_ch, n_times)) * 1.0
epoch = gains[:, None] * source[None, :] + noise
epoch_tuple = (epoch.astype(np.float64), CH_NAMES_19, BASE_CFG, gains, source)

print(f'synthetic_epoch: epoch shape = {epoch_tuple[0].shape}')
print(f'synthetic_epoch: ch_names len = {len(epoch_tuple[1])}')
print(f'synthetic_epoch: gains shape = {epoch_tuple[3].shape}')
print(f'synthetic_epoch: source shape = {epoch_tuple[4].shape}')

# Test synthetic_epochs_batch shape
rng2 = np.random.default_rng(99)
n_epochs, n_ch, n_times = 5, 19, 1000
source2 = rng2.standard_normal((n_epochs, n_times)) * 50.0
gains2 = rng2.uniform(0.5, 1.0, n_ch)
noise2 = rng2.standard_normal((n_epochs, n_ch, n_times)) * 1.0
epochs2 = gains2[None, :, None] * source2[:, None, :] + noise2
epochs_tuple = (epochs2.astype(np.float64), CH_NAMES_19, BASE_CFG)

print(f'synthetic_epochs_batch: epochs shape = {epochs_tuple[0].shape}')
print(f'synthetic_epochs_batch: ch_names len = {len(epochs_tuple[1])}')
