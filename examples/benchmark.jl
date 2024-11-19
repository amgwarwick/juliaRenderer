import MuJoCo
import Images
include("../src/juliaRenderer.jl")

model_path = "examples/rodent_with_floor.xml"
model = MuJoCo.load_model(model_path)

function benchmark(model; n_envs=64, res=64, n_iter=100)
    datas = [MuJoCo.init_data(model) for _ = 1:n_envs]
    for data in datas
        MuJoCo.step!(model, data)
    end
    batchRenderer = juliaRenderer.BatchRenderer(model; n_envs, res)
    GC.gc()
    start_t = time()
    for i=1:n_iter
        image_data = juliaRenderer.render(batchRenderer, datas)
    end
    stop_t = time()
    juliaRenderer.EGLUtils.free!(batchRenderer.egl_resources)
    return n_envs*n_iter/(stop_t - start_t)
end

for res = 2 .^ (5:11)
    for n_envs = 2 .^ (1:11)
        fps = benchmark(model; n_envs=n_envs, res=res, n_iter=100)
        println("$n_envs,$res,$fps")
    end
end
