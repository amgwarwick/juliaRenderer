import MuJoCo
using Images
include("renderer.jl")
n_envs = 16
model_path = "rodent_with_floor.xml"
model = MuJoCo.load_model(model_path)

datas = [MuJoCo.init_data(model) for _ = 1:n_envs]
for data in datas
    MuJoCo.step!(model, data)
end
batchRenderer = Renderer.BatchRenderer(model, res=512, n_envs=n_envs)
image_data = Renderer.render(batchRenderer, datas)

float_image_data = convert(Array{Float32, 3}, image_data) / 255.0
img = collect(colorview(RGB, float_image_data)')
save("test1.png", img)
