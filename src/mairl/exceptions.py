"""Package-specific failures."""


class MAIRLError(Exception):
    """Base class for expected MAIRL failures."""


class ConfigurationError(MAIRLError, ValueError):
    """A configuration is internally inconsistent."""


class CompilationError(MAIRLError, ValueError):
    """A static problem failed validation or compilation."""


class ConvergenceError(MAIRLError, RuntimeError):
    """A numerical solver did not meet its declared convergence checks."""
