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
    ref = ref_image(model, res)
    batchRenderer = juliaRenderer.BatchRenderer(model; n_envs, res)
    total_error = validate_against_ref(batchRenderer, datas, ref)
    GC.gc()
    start_t = time()
    for i=1:n_iter
        image_data = juliaRenderer.render(batchRenderer, datas)
    end
    stop_t = time()
    juliaRenderer.EGLUtils.free!(batchRenderer.egl_resources)
    return n_envs*n_iter/(stop_t - start_t), total_error
end

function ref_image(model, res)
    data = MuJoCo.init_data(model)
    MuJoCo.step!(model, data)
    batchRenderer = juliaRenderer.BatchRenderer(model; n_envs=1, res)
    ref_im = juliaRenderer.render(batchRenderer, [data])
    juliaRenderer.EGLUtils.free!(batchRenderer.egl_resources)
    return ref_im
end

function validate_against_ref(batchRenderer, datas, ref)
    image_data = juliaRenderer.render(batchRenderer, datas)
    res = batchRenderer.res
    total_error = 0.0
    for i=1:batchRenderer.n_envs
        sub_image = (@view image_data[:, (res*(i-1)+1):res*i, :])
        total_error += norm(sub_image - ref)
    end
    return total_error
end

println("n_envs,res,fps,total_error")
for res = 2 .^ (5:11)
    for n_envs = 2 .^ (1:11)
        fps,total_error = benchmark(model; n_envs=n_envs, res=res, n_iter=100)
        println("$n_envs,$res,$fps,$total_error")
    end
end
