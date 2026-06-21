"""SO-101 canonical sim/real parity 실행 계층.

이 패키지는 Isaac Sim, ROS 2, LeRobot을 import하지 않는다. 각 런타임 adapter는
경계에서만 이 패키지의 계약·codec·executor를 사용한다.
"""

from .calibration import CalibrationBundle, CalibrationError, MonotonePchip
from .contract import CANONICAL_SCHEMA, PolicyIOContract
from .executor import (
    Chunk,
    MotionLimiter,
    SingleFlightChunkExecutor,
    SingleFlightInferenceWorker,
    prefetch_lead_from_p99,
)
from .manifest import RuntimeManifest, RuntimeManifestError
from .lease import LeaseError, MotionLease
from .model_codec import ModelCodec, ModelCodecError
from .runtime import (
    CanonicalObservation,
    CanonicalRuntime,
    RuntimeAdapter,
    RuntimeHashes,
)
from .motor_profile import verify_motor_profile

__all__ = [
    "CANONICAL_SCHEMA",
    "CalibrationBundle",
    "CalibrationError",
    "Chunk",
    "MonotonePchip",
    "MotionLimiter",
    "MotionLease",
    "LeaseError",
    "ModelCodec",
    "ModelCodecError",
    "PolicyIOContract",
    "RuntimeManifest",
    "RuntimeManifestError",
    "SingleFlightChunkExecutor",
    "SingleFlightInferenceWorker",
    "prefetch_lead_from_p99",
]
