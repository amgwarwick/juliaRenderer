struct GeomRenderer
    instance_vbo::UInt32
    shader_program::Int32
    sizeindices::Int32
    vao::UInt32
    shader_locations::Dict{String, Int32}
end

function render_geoms(typeRenderer::GeomRenderer, batchRenderer::BatchRenderer, instance_data, light_data_xdir, light_data_xpos, camera_data_mat, camera_data_pos)
    final, forward = lookatmatrix(camera_data_pos, camera_data_mat)
    glBindBuffer(GL_ARRAY_BUFFER, typeRenderer.instance_vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(instance_data), instance_data, GL_DYNAMIC_DRAW)

    glUseProgram(typeRenderer.shader_program)

    glUniformMatrix4fv(typeRenderer.shader_locations["view"], 1, GL_FALSE, final)
    glUniformMatrix4fv(typeRenderer.shader_locations["projection"], 1, GL_FALSE, batchRenderer.projection_matrix)
    
    glUniform3fv(typeRenderer.shader_locations["lightPos"], 1, light_data_xpos)
    glUniform3fv(typeRenderer.shader_locations["lightDir"], 1, light_data_xdir)
    glUniform3fv(typeRenderer.shader_locations["ambient"], 1, batchRenderer.light_model["light_ambient"])
    glUniform3fv(typeRenderer.shader_locations["diffuse"], 1, batchRenderer.light_model["light_diffuse"])
    glUniform3fv(typeRenderer.shader_locations["specular"], 1, batchRenderer.light_model["light_specular"])
    glUniform1f(typeRenderer.shader_locations["cutoff"], deg2rad(batchRenderer.light_model["light_cutoff"]))  
    glUniform1f(typeRenderer.shader_locations["exponent"], batchRenderer.light_model["light_exponent"])  
    glUniform1i(typeRenderer.shader_locations["directional"], batchRenderer.light_model["light_directional"])  
    glUniform3fv(typeRenderer.shader_locations["attenuation"], 1, batchRenderer.light_model["light_attenuation"])
    glUniform3fv(typeRenderer.shader_locations["viewPos"], 1, camera_data_pos)
    glUniform3fv(typeRenderer.shader_locations["headlightDir"], 1, forward)
    glUniform1f(typeRenderer.shader_locations["n_env"], batchRenderer.n_envs) 
    glUniform1f(typeRenderer.shader_locations["res"], batchRenderer.res)  
    
    glBindVertexArray(typeRenderer.vao)
    
    if typeRenderer.sizeindices == 0
        glDrawArraysInstanced(GL_TRIANGLES, 0, 36, size(instance_data, 2))
    else
        glDrawElementsInstanced(GL_TRIANGLES, typeRenderer.sizeindices, GL_UNSIGNED_INT, C_NULL, size(instance_data, 2))
    end
end

function BoxRenderer()
    vertices, indices, normals, tex_coords = generate_box_vertices()
    return GeomRenderer(vertices, indices, normals, tex_coords)
end

function SphereRenderer()
    vertices, indices, normals, tex_coords = generate_sphere_vertices(1.0)
    return GeomRenderer(vertices, indices, normals, tex_coords)
end

function CapsuleRenderer()
    vertices, indices, normals, tex_coords = generate_capsule_vertices(1.0, 1.0, 36, 18)
    return GeomRenderer(vertices, indices, normals, tex_coords)
end

function PlaneRenderer()
    vertices, indices, normals, tex_coords = generate_plane_vertices(Float32(1.0), Float32(1.0))
    return GeomRenderer(vertices, indices, normals, tex_coords)
end

function GeomRenderer(vertices, indices, normals, tex_coords)
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
    
    shader_locations = Dict(
        "projection" => glGetUniformLocation(shader_program, "projection"),
        "lightPos" => glGetUniformLocation(shader_program, "lightPos"),
        "lightDir" => glGetUniformLocation(shader_program, "lightDir"),
        "ambient" => glGetUniformLocation(shader_program, "ambient"),
        "diffuse" => glGetUniformLocation(shader_program, "diffuse"),
        "specular" => glGetUniformLocation(shader_program, "specular"),
        "cutoff" => glGetUniformLocation(shader_program, "cutoff"),
        "exponent" => glGetUniformLocation(shader_program, "exponent"),
        "directional" => glGetUniformLocation(shader_program, "directional"),
        "attenuation" => glGetUniformLocation(shader_program, "attenuation"),
        "viewPos" => glGetUniformLocation(shader_program, "viewPos"),
        "headlightDir" => glGetUniformLocation(shader_program, "headlightDir"),
        "view" => glGetUniformLocation(shader_program, "view"),
        "n_env" => glGetUniformLocation(shader_program, "n_env"),
        "res" => glGetUniformLocation(shader_program, "res")
    )

    return GeomRenderer(instance_vbo[], shader_program, indices === nothing ? 0 : size(indices ,1), vao[], shader_locations)

end
