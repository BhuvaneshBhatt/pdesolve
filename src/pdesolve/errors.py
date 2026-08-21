from __future__ import annotations


class PDEError(Exception):
    """Base class for errors raised by PDESolve."""


class PDEInputError(PDEError, ValueError):
    """The supplied PDE problem or conditions are invalid."""


class PDEMethodNotApplicable(PDEError, NotImplementedError):
    """A solver method cannot be applied to the current problem."""


class PDETransformationError(PDEError):
    """A symbolic transformation failed after a method was selected."""


class PDEVerificationError(PDEError):
    """Verification could not be completed because of an internal error."""


class PDESolveError(PDEError):
    """No planned solver produced a solution."""


__all__ = [
    "PDEError",
    "PDEInputError",
    "PDEMethodNotApplicable",
    "PDETransformationError",
    "PDEVerificationError",
    "PDESolveError",
]
