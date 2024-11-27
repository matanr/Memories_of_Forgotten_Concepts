from enum import Enum


class EnumBase(Enum):
    @classmethod
    def get_all_types_names(cls):
        return [d.name for d in list(cls)]

    @classmethod
    def get_all_types(cls):
        return [d for d in list(cls)]

    @classmethod
    def name2value(cls, name):
        for d in list(cls):
            if d.name.lower() == name.lower():
                return d
        raise ValueError(f"Invalid name {name}")

    @classmethod
    def get_associated_type(cls, val):
        if isinstance(val, int) or isinstance(val, float) or (isinstance(val, str) and val.isdigit()):
            return cls(int(val))
        elif isinstance(val, str):
            return cls.name2value(val)
        else:
            raise ValueError(f"Invalid type {type(val)}")


class ModelType(EnumBase):
    EraseDiff = 1
    ESD = 2
    FMN = 3
    Salun = 4
    Scissorhands = 5
    SPM = 6
    UCE = 7


class AttackType(EnumBase):
    NTI = 1


class DataType(EnumBase):
    Nudity = 1
    Object = 2
    VanGogh = 3
