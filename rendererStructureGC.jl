using MuJoCo
using Libdl
using Base.Libc: Ptr
using LinearAlgebra
using GLM
using ModernGL, GLAbstraction
using FileIO
using Images
using ColorTypes
using LinearAlgebra
using GeometryTypes
using Colors
include("basic_geoms.jl")
include("extract_data.jl")
include("opengl_utils.jl")

struct GeomRenderer
    instance_vbo::UInt32
    shader_program::Int32
    sizeindices::Int32
    vao::UInt32
    shader_locations::Dict{String, Int32}
end

struct BatchRenderer
    res::Int32
    n_envs::Int32

    egl_ctx_success::Int32

    projection_matrix::Matrix{Float32}

    light_model::Dict{String, Any}
    #textures when ready

    n_geoms_by_type::Dict{String, Int32}

    renderers::Vector{GeomRenderer}
end
include("geom_renderer.jl")

function BatchRenderer(model; res, n_envs)

    data = MuJoCo.init_data(model)

	#scene inside
    scn = MuJoCo.VisualiserScene()
    cam = MuJoCo.VisualiserCamera()
    opt = MuJoCo.VisualiserOption()
    pert = MuJoCo.VisualiserPerturb()
    MuJoCo.mjv_makeScene(model, scn, 1000) #scn should now contain the scene object

    # mjCAT_ALL = 7
    catmask = 7

    mjv_updateScene(model, data, opt, pert, cam, catmask, scn)

    n_geoms = scn.ngeom

    egl_ctx_success = setup_egl(Cint(n_envs*res), Cint(res))

    camera_fovy = extract_camera_model(model)

    projection_matrix = Float32.(collect(transpose(perspective(Float32(deg2rad(camera_fovy)), Float32(1.0), Float32(0.1), Float32(1000.0)))))

    light_ambient, light_attenuation, light_cutoff, light_diffuse, light_dir, light_directional, light_exponent, light_pos, light_specular = extract_light_model(model)
    
    light_model = Dict(
        "light_ambient" => light_ambient,
        "light_attenuation" => light_attenuation,
        "light_cutoff" => light_cutoff,
        "light_diffuse" => light_diffuse,
        "light_dir" => light_dir,
        "light_directional" => light_directional,
        "light_exponent" => light_exponent,
        "light_pos" => light_pos,
        "light_specular" => light_specular
    )

    nboxes, nspheres, nplanes, ncapsules = count_geoms(scn.geoms, n_geoms) # pass model only

    # Create the dictionary with values cast to Int32
    n_geoms_by_type = Dict(
        "nboxes" => Int32(nboxes),
        "nspheres" => Int32(nspheres),
        "nplanes" => Int32(nplanes),
        "ncapsules" => Int32(ncapsules)
    )

    #initialise vector of renderers here
    boxRenderer = BoxRenderer()
    sphereRenderer = SphereRenderer()
    capsuleRenderer = CapsuleRenderer()
    planeRenderer = PlaneRenderer()
    
    renderers = [boxRenderer, sphereRenderer, capsuleRenderer, planeRenderer] 

    #display.(renderers)

    return BatchRenderer(res, n_envs, egl_ctx_success, projection_matrix, light_model, n_geoms_by_type, renderers)
end

function render(batchRenderer, datas)

    instance_data_boxes, instance_data_spheres, instance_data_planes,  instance_data_capsules = extract_geom_data(model, datas, batchRenderer.n_envs, batchRenderer.n_geoms_by_type)

    camera_data_pos, camera_data_mat = extract_camera_data(datas[1]) #will change
    camera_data_pos = Float32.(camera_data_pos)

    light_data_xdir, light_data_xpos = extract_light_data(datas[1], 1) #will change

    glEnable(GL_DEPTH_TEST)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    render_geoms(batchRenderer.renderers[1], batchRenderer, instance_data_boxes, light_data_xdir, light_data_xpos, camera_data_mat, camera_data_pos)
    render_geoms(batchRenderer.renderers[2], batchRenderer, instance_data_spheres, light_data_xdir, light_data_xpos, camera_data_mat, camera_data_pos)
    render_geoms(batchRenderer.renderers[3], batchRenderer, instance_data_capsules, light_data_xdir, light_data_xpos, camera_data_mat, camera_data_pos)
    render_geoms(batchRenderer.renderers[4], batchRenderer, instance_data_planes, light_data_xdir, light_data_xpos, camera_data_mat, camera_data_pos)

    return save_egl_image("rendered_image_julia.png", batchRenderer.n_envs*batchRenderer.res, batchRenderer.res)
    
end
