using MuJoCo
using Libdl
using Base.Libc: Ptr
using LinearAlgebra
using GLM
using ModernGL, GLAbstraction
using FileIO
import Images
using ColorTypes
using LinearAlgebra
using GeometryTypes
using Colors
using ProgressMeter
using Profile

include("rendererStructureFast.jl")

n_envs = 16
model_path = "rodent_with_floor.xml"
model = MuJoCo.load_model(model_path)

datas = [MuJoCo.init_data(model) for _ = 1:n_envs]
for data in datas
    MuJoCo.step!(model, data)
end
batchRenderer = BatchRenderer(model, res=64, n_envs=n_envs)

function benchMark(batchRenderer, datas)
    for t = 1:32
        images = render(batchRenderer, datas)
    end
end

@time benchMark(batchRenderer, datas)
