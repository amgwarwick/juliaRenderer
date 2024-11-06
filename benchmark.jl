import MuJoCo
include("renderer.jl")
n_envs = 2
model_path = "rodent_with_floor.xml"
model = MuJoCo.load_model(model_path)

datas = [MuJoCo.init_data(model) for _ = 1:n_envs]
for data in datas
    MuJoCo.step!(model, data)
end
batchRenderer = Renderer.BatchRenderer(model, res=256, n_envs=n_envs)
Renderer.render(batchRenderer, datas)
Renderer.save_egl_image("test.png", Int32(256*n_envs), Int32(256))
#Renderer.to_cpu_array(Int32(64*n_envs), Int32(64))

function benchmark(batchRenderer, datas)
    for t = 1:32
        images = Renderer.render(batchRenderer, datas)
    end
end

@time benchmark(batchRenderer, datas)