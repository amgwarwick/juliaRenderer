struct BatchRenderer
    model::MuJoCo.Model
    res::Int32
    n_envs::Int32

    projection_matrix::Matrix{Float32}

    light_model::Dict{String, Any}
    #textures when ready

    geom_counts::Dict{MuJoCo.mjtGeom, Int32}
    instance_data::Dict{MuJoCo.mjtGeom, Matrix{Float32}}
    pixel_buffer::Vector{UInt8}

    renderers::Vector{GeomRenderer}
    shader_program::Int32
    shader_locations::Dict{String, Int32}

    texture_array::UInt32

    egl_resources::EGLUtils.EGLResources
end

function BatchRenderer(model; res, n_envs)
#n_envs = 1; res = 64; model = Main.model
    egl_resources = EGLUtils.init_egl(n_envs*res, res)

    data = MuJoCo.init_data(model)

	#scene inside
    scn = MuJoCo.VisualiserScene()
    cam = MuJoCo.VisualiserCamera()
    opt = MuJoCo.VisualiserOption()
    pert = MuJoCo.VisualiserPerturb()
    MuJoCo.mjv_makeScene(model, scn, 1000) #scn should now contain the scene object

    mjv_updateScene(model, data, opt, pert, cam, MuJoCo.mjCAT_ALL, scn)
    camera_fovy = extract_camera_model(model)
    projection_matrix = Float32.(collect(transpose(perspective(Float32(deg2rad(camera_fovy)), Float32(1.0), Float32(0.1), Float32(1000.0)))))
    light_model = extract_light_model(model)
    geom_counts = count_geoms(scn)
    supported_geoms = (MuJoCo.mjGEOM_PLANE, MuJoCo.mjGEOM_SPHERE,
                       MuJoCo.mjGEOM_CAPSULE, MuJoCo.mjGEOM_BOX)
    instance_data = Dict(geom_id => zeros(Float32, 22, n_envs * geom_counts[geom_id])
                         for geom_id in supported_geoms)

    boxRenderer = BoxRenderer()
    sphereRenderer = SphereRenderer()
    capsuleRenderer = CapsuleRenderer()
    planeRenderer = PlaneRenderer()
    
    geom_renderers = [boxRenderer, sphereRenderer, capsuleRenderer, planeRenderer] 

    shader_program = compile_shaders()
    shader_variables = ["projection", "lightPos", "lightDir", "ambient", "diffuse",
                        "specular", "cutoff", "exponent", "directional", "attenuation",
                        "viewPos", "headlightDir", "view", "n_env", "res", "texArray"]
    shader_locations = Dict(v => glGetUniformLocation(shader_program, v) for v in shader_variables)

    pixel_buffer = Vector{UInt8}(undef, n_envs * res * res * 3)

    texture_array_pointer = upload_textures(extract_textures(model))

    return BatchRenderer(model, res, n_envs, projection_matrix, light_model, geom_counts,
                         instance_data, pixel_buffer, geom_renderers, shader_program,
                         shader_locations, texture_array_pointer, egl_resources)
end

function render!(batchRenderer, datas)
    extract_geom_data!(batchRenderer, datas)

    camera_data_pos, camera_data_mat = extract_camera_data(datas[1]) #will change
    camera_data_pos = Float32.(camera_data_pos)
    final, forward = lookatmatrix(camera_data_pos, camera_data_mat)

    light_data_xdir, light_data_xpos = extract_light_data(datas[1], 1) #will change

    glEnable(GL_DEPTH_TEST)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(batchRenderer.shader_program)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D_ARRAY, batchRenderer.texture_array)
    glUniform1i(batchRenderer.shader_locations["texArray"], 0)
    
    glUniformMatrix4fv(batchRenderer.shader_locations["view"], 1, GL_FALSE, final)
    glUniformMatrix4fv(batchRenderer.shader_locations["projection"], 1, GL_FALSE, batchRenderer.projection_matrix)
    
    glUniform3fv(batchRenderer.shader_locations["lightPos"], 1, light_data_xpos)
    glUniform3fv(batchRenderer.shader_locations["lightDir"], 1, light_data_xdir)
    glUniform3fv(batchRenderer.shader_locations["ambient"], 1, batchRenderer.light_model["light_ambient"])
    glUniform3fv(batchRenderer.shader_locations["diffuse"], 1, batchRenderer.light_model["light_diffuse"])
    glUniform3fv(batchRenderer.shader_locations["specular"], 1, batchRenderer.light_model["light_specular"])
    glUniform1f(batchRenderer.shader_locations["cutoff"], deg2rad(batchRenderer.light_model["light_cutoff"]))  
    glUniform1f(batchRenderer.shader_locations["exponent"], batchRenderer.light_model["light_exponent"])  
    glUniform1i(batchRenderer.shader_locations["directional"], batchRenderer.light_model["light_directional"])  
    glUniform3fv(batchRenderer.shader_locations["attenuation"], 1, batchRenderer.light_model["light_attenuation"])
    glUniform3fv(batchRenderer.shader_locations["viewPos"], 1, camera_data_pos)
    glUniform3fv(batchRenderer.shader_locations["headlightDir"], 1, forward)
    glUniform1f(batchRenderer.shader_locations["n_env"], batchRenderer.n_envs)
    glUniform1f(batchRenderer.shader_locations["res"], batchRenderer.res)

    render_geoms(batchRenderer.renderers[1], batchRenderer.instance_data[MuJoCo.mjGEOM_BOX])
    render_geoms(batchRenderer.renderers[2], batchRenderer.instance_data[MuJoCo.mjGEOM_SPHERE])
    render_geoms(batchRenderer.renderers[3], batchRenderer.instance_data[MuJoCo.mjGEOM_CAPSULE])
    render_geoms(batchRenderer.renderers[4], batchRenderer.instance_data[MuJoCo.mjGEOM_PLANE])

    glReadBuffer(GL_FRONT)
    height = batchRenderer.res
    width = batchRenderer.n_envs * height
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, batchRenderer.pixel_buffer)
	return reshape(batchRenderer.pixel_buffer, (3, width, height))
end
