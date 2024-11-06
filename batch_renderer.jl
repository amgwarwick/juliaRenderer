struct BatchRenderer
    model::MuJoCo.Model
    res::Int32
    n_envs::Int32
    projection_matrix::Matrix{Float32}
    light_model::Dict{String, Any}
    #textures when ready
    geom_counts::Dict{MuJoCo.mjtGeom, Int32}
    renderers::Vector{GeomRenderer}
    shader_program::Int32
    shader_locations::Dict{String, Int32}
end

function BatchRenderer(model; res, n_envs)
    setup_egl(Cint(n_envs*res), Cint(res))

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
    geom_counts = count_geoms(scn)#scn.geoms, n_geoms) # pass model only

    boxRenderer = BoxRenderer()
    sphereRenderer = SphereRenderer()
    capsuleRenderer = CapsuleRenderer()
    planeRenderer = PlaneRenderer()
    
    geom_renderers = [boxRenderer, sphereRenderer, capsuleRenderer, planeRenderer] 

    shader_program = compile_shaders()
    shader_variables = ["projection", "lightPos", "lightDir", "ambient", "diffuse",
                        "specular", "cutoff", "exponent", "directional", "attenuation",
                        "viewPos", "headlightDir", "view", "n_env", "res"]
    shader_locations = Dict(v => glGetUniformLocation(shader_program, v) for v in shader_variables)
    return BatchRenderer(model, res, n_envs, projection_matrix, light_model, 
                         geom_counts, geom_renderers, shader_program, shader_locations)
end

function render(batchRenderer, datas)
    instance_data = extract_geom_data(batchRenderer, datas)

    camera_data_pos, camera_data_mat = extract_camera_data(datas[1]) #will change
    camera_data_pos = Float32.(camera_data_pos)
    final, forward = lookatmatrix(camera_data_pos, camera_data_mat)

    light_data_xdir, light_data_xpos = extract_light_data(datas[1], 1) #will change

    glEnable(GL_DEPTH_TEST)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(batchRenderer.shader_program)
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
    
    render_geoms(batchRenderer.renderers[1], instance_data[MuJoCo.mjGEOM_BOX])
    render_geoms(batchRenderer.renderers[2], instance_data[MuJoCo.mjGEOM_SPHERE])
    render_geoms(batchRenderer.renderers[3], instance_data[MuJoCo.mjGEOM_CAPSULE])
    render_geoms(batchRenderer.renderers[4], instance_data[MuJoCo.mjGEOM_PLANE])

    #return save_egl_image("rendered_image_julia.png", BatchRenderer.n_envs*BatchRenderer.res, BatchRenderer.res)
end





