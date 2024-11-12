import MuJoCo
include("renderer.jl")
n_envs = 16
model_path = "rodent_with_floor.xml"
model = MuJoCo.load_model(model_path)

datas = [MuJoCo.init_data(model) for _ = 1:n_envs]
for data in datas
    MuJoCo.step!(model, data)
end
batchRenderer = Renderer.BatchRenderer(model, res=512, n_envs=n_envs)
Renderer.render(batchRenderer, datas)
