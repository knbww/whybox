from .base import Annotation, Domain, DomainSpec, FactorSpec, StateBatch, annotate, sigmoid
from .agreement import Agreement
from .car import Car
from .clinic import Clinic
from .interact import Interact
from .langtasks import Entail, Polarity
from .seqworld import SeqWorld
from .skirmish import Skirmish
from .wikifacts import WikiFacts
from .witness import Witness

REGISTRY = {"skirmish": Skirmish, "clinic": Clinic, "car": Car, "seqworld": SeqWorld,
            "agreement": Agreement, "polarity": Polarity, "entail": Entail,
            "interact": Interact, "witness": Witness,
            "wikifacts": WikiFacts}


def get_domain(name: str) -> Domain:
    if name not in REGISTRY:
        raise KeyError(f"unknown domain {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name]()


__all__ = [
    "Agreement", "Annotation", "Car", "Clinic", "Entail", "Interact", "Polarity", "Domain", "DomainSpec", "FactorSpec", "REGISTRY",
    "SeqWorld", "Skirmish", "StateBatch", "WikiFacts", "Witness", "annotate", "get_domain", "sigmoid",
]
