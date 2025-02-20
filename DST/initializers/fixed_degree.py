import torch


from DST.initializers.grand import SparseInitializer


class FixedDegreeRandomInitializer(SparseInitializer):
    @classmethod
    def initialize(
        cls,
        input_shape: int,
        output_shape: int,
        device="cpu",
        sparsity=0.9,
        built_edges: torch.Tensor = torch.empty((0, 2)),
    ) -> torch.Tensor:
        nz = int((1 - sparsity) * input_shape * output_shape)
        degree = nz // input_shape
        res = torch.zeros((nz, 2), dtype=torch.int32)
        if built_edges.size(0) > 0:
            raise NotImplementedError("This part isnt implemented")
        for i in range(input_shape):
            res[i * degree : i * degree + degree, 0] = i
            res[i * degree : i * degree + degree, 1] = torch.randperm(output_shape)[:degree]
        return res.to(device)

