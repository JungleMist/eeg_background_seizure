from eeg_bg.config.settings import load_config
from eeg_bg.io.cache import load_or_compute, make_cache_key
from eeg_bg.io.annotation import extract_bckg_intervals
from eeg_bg.io.edf_reader import load_edf
from eeg_bg.io.dataset import build_subject_index, assign_splits
from eeg_bg.preprocessing.epoch import bandpass_filter, slice_epochs
from eeg_bg.preprocessing.reference import detect_reference, filter_by_reference
from eeg_bg.decomposition.wiener import decompose_epoch, decompose_subject, WienerResult
from eeg_bg.decomposition.wiener_scalar import decompose_epoch as scalar_decompose
from eeg_bg.decomposition.ica import fit_ica, apply_ica
from eeg_bg.verification.coherence import compute_pairwise_coherence, run_v1
from eeg_bg.verification.transitivity import run_v2, run_v3
from eeg_bg.visualization.filter_plots import plot_wiener_filter_response, plot_all_pairs_response
from eeg_bg.visualization.coherence_plots import plot_coherence_matrix, plot_coherence_reduction, plot_signal_decomposition
from eeg_bg.visualization.verification_plots import plot_v2_transitivity, plot_v3_frequency_variation, plot_ica_vs_wiener_coherence
print('All imports OK')
