"""User-facing ECMAD product naming for the desktop application."""

PRODUCT_ACRONYM = "ECMAD"
PRODUCT_NAME = "EEG Channel Matrix Adaptive Denoiser"
APPLICATION_NAME = f"{PRODUCT_ACRONYM} Studio"
WINDOW_TITLE = f"{APPLICATION_NAME} — {PRODUCT_NAME}"

# Keep the existing settings namespace so upgrading the branded application does
# not discard a user's saved preview, batch, and artifact-threshold preferences.
SETTINGS_ORGANIZATION = "eeg_bg"
SETTINGS_APPLICATION = "eeg_bg Studio"
