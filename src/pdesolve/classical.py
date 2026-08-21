"""Classical PDE solvers and supporting symbolic utilities."""

from .classical_methods import *
from .results import *
from .problem import *
from .classification import *
from .transforms import *
from .conservation_laws import *
from .ivp_bvp import *
from .operator_symbol import *
from .constant_coeff import *
from .first_order_nonlinear import *
from .unified_transform import *
from .hyperbolic_system import *
from .benchmark_helpers import *
from .family_recognizers import *
from .complete_integral_helpers import *
from .dispatcher import (
    extract_solution_trace as extract_solution_trace,
    pdesolve as pdesolve,
    summarize_solution_record as summarize_solution_record,
)

__all__ = [name for name in globals() if not name.startswith("_")]
