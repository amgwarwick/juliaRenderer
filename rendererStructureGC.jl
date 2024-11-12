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

function setup_opengl(vertices, normals, tex_coords, indices)

	vao = Ref(GLuint(0))
    glGenVertexArrays(1, vao)
    
	vbo = Ref(GLuint(0))
    glGenBuffers(1, vbo)
	
	normal_vbo = Ref(GLuint(0))
	glGenBuffers(1, normal_vbo)
	
	
	tex_coord_vbo = Ref(GLuint(0))
	glGenBuffers(1, tex_coord_vbo)
	
	instance_vbo = Ref(GLuint(0))
	glGenBuffers(1, instance_vbo)
    

    glBindVertexArray(vao[])
    


    glBindBuffer(GL_ARRAY_BUFFER, vbo[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), C_NULL)    
    glEnableVertexAttribArray(0)



    # Bind normal data
    glBindBuffer(GL_ARRAY_BUFFER, normal_vbo[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(normals), normals, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), C_NULL)
    glEnableVertexAttribArray(1)


    # Bind texture coordinate data
    glBindBuffer(GL_ARRAY_BUFFER, tex_coord_vbo[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(tex_coords), tex_coords, GL_STATIC_DRAW)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), C_NULL)
    glEnableVertexAttribArray(2)


    # Optionally bind EBO
    if indices !== nothing
    	ebo = Ref(GLuint(0))
        glGenBuffers(1, ebo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo[])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW)
    else
    	ebo = nothing
    end
    

    # Bind and configure instance VBO
    glBindBuffer(GL_ARRAY_BUFFER, instance_vbo[])
    
    # Define instance attribute pointers
    for i in 0:3
        glVertexAttribPointer(3 + i, 4, GL_FLOAT, GL_FALSE, 22 * sizeof(GL_FLOAT), Ptr{Cvoid}(i * 4 * sizeof(GLfloat)))
        glEnableVertexAttribArray(3 + i)
        glVertexAttribDivisor(3 + i, 1)
    end

    glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, 22 * sizeof(GL_FLOAT), Ptr{Cvoid}(16 * sizeof(GL_FLOAT)))
    glEnableVertexAttribArray(7)
    glVertexAttribDivisor(7, 1)

    glVertexAttribPointer(8, 1, GL_FLOAT, GL_FALSE, 22 * sizeof(GL_FLOAT), Ptr{Cvoid}(20 * sizeof(GL_FLOAT)))
    glEnableVertexAttribArray(8)
    glVertexAttribDivisor(8, 1)

    glVertexAttribPointer(9, 1, GL_FLOAT, GL_FALSE, 22 * sizeof(GL_FLOAT), Ptr{Cvoid}(21 * sizeof(GL_FLOAT)))
    glEnableVertexAttribArray(9)
    glVertexAttribDivisor(9, 1)

    glBindVertexArray(vao[])
    
    shader_program = compile_shaders()
    
    if indices === nothing
    	return vao[], vbo[], normal_vbo[], tex_coord_vbo[], instance_vbo[], ebo, shader_program
    else
    	return vao[], vbo[], normal_vbo[], tex_coord_vbo[], instance_vbo[], ebo[], shader_program
    end
end

function setup_data(instance_data, instance_vbo)
    # Bind and update instance data (transformation matrices + rgba)
    glBindBuffer(GL_ARRAY_BUFFER, instance_vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(instance_data), instance_data, GL_DYNAMIC_DRAW)
end
