from .passage import Passage
from .rotor import Rotor
from .conduit import Conduit
from .pusher import Pusher
from .chroma import Chroma

REGISTRY = {cls.env_id: cls for cls in (Passage, Rotor, Conduit, Pusher, Chroma)}


def make(env_id: str, seed: int = 0):
    return REGISTRY[env_id](seed=seed)
