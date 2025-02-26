from DST.layers.Snack import Snack
from DST.initializers import FixedDegreeRandomInitializer, UniformInitializer
from DST.pruner_grower import ZetaPrunerGrower


def tst1(sparsity):
    input_size = 1000
    output_size = 1000
    sn = Snack(
        input_size,
        output_size,
        sparsity=sparsity,
        initializer=FixedDegreeRandomInitializer,
        device="cuda",
    )
    print(sn.indices_a.shape)
    pruner_grower = ZetaPrunerGrower(sn, zeta=0.1)
    pruner_grower.prune()
    print(sn.indices_a.shape)
    pruner_grower.regrow(init=UniformInitializer, device="cuda")
    print(sn.indices_a.shape)


def tst2():
    input_size = 2
    output_size = 2
    sn = Snack(
        input_size,
        output_size,
        sparsity=0.5,
        initializer=FixedDegreeRandomInitializer,
        device="cuda",
    )
    print(sn.values)
    pruner_grower = ZetaPrunerGrower(sn, zeta=0.5)
    pruner_grower.prune()
    print(sn.values)
    pruner_grower.regrow(init=UniformInitializer, device="cuda")
    print(sn.values)


if __name__ == "__main__":
    tst1(sparsity=0.21)
    # tst2()
