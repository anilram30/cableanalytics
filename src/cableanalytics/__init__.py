"""cableanalytics - loss, temperature and production analytics for high-frequency cables."""
__version__ = "0.1.0"
__author__ = "Sreeram Anil"
from .correlate import fit_correlation, predict_coefficients
from .derating import DeratingModel, fit_derating
from .lossfit import LossFit, fit_loss
from .periodic import attribute, period_from_profile, period_from_return_loss
from .physics import Design, hf_properties
from .predict import predict_performance

__all__ = ["fit_loss", "LossFit", "fit_derating", "DeratingModel", "period_from_return_loss", "period_from_profile",
           "attribute", "fit_correlation", "predict_coefficients", "predict_performance", "Design", "hf_properties", "__version__"]
