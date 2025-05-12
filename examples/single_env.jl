import MuJoCo
import Images
import LinearAlgebra: norm
include("../src/juliaRenderer.jl")

model_path = "examples/rodent_with_floor.xml"
model = MuJoCo.load_model(model_path)

res = 512
batchRenderer = juliaRenderer.BatchRenderer(model; n_envs=1, res)
data = MuJoCo.init_data(model)
MuJoCo.forward!(model, data)
frame = juliaRenderer.render!(batchRenderer, [data])
Images.colorview(Images.RGB, frame ./ 255.0f0)'[end:-1:1, :]
#all(im .== 0)

#Images.colorview(Images.RGBA, juliaRenderer.extract_textures(model)[:, :, :, 3] ./ 255.0f0)