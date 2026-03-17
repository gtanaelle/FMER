from relax.env.vector.base import VectorEnv
from relax.env.vector.serial import SerialVectorEnv
from relax.env.vector.process0 import ProcessVectorEnv as GymProcessVectorEnv
from relax.env.vector.process1 import ProcessVectorEnv as PipeProcessVectorEnv

try:
    from relax.env.vector.process2 import ProcessVectorEnv as SpinlockProcessVectorEnv
except ImportError:
    SpinlockProcessVectorEnv = None

try:
    from relax.env.vector.process3 import ProcessVectorEnv as FutexProcessVectorEnv
except ImportError:
    FutexProcessVectorEnv = None
