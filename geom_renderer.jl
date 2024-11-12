struct GeomRenderer
    instance_vbo::UInt32
    shader_program::Int32
    sizeindices::Int32
    vao::UInt32
    shader_locations::Dict{String, Int32}
end

function display(typeRenderer::GeomRenderer, instance_data, light_data_xdir, light_data_xpos, camera_data_mat, camera_data_pos, BatchRenderer::BatchRenderer)
	    
    final, forward = lookatmatrix(camera_data_pos, camera_data_mat)

    setup_data(instance_data, typeRenderer.instance_vbo)

    glUseProgram(typeRenderer.shader_program)

    glUniformMatrix4fv(typeRenderer.shader_locations["view_loc"], 1, GL_FALSE, final)
    glUniformMatrix4fv(typeRenderer.shader_locations["projection_loc"], 1, GL_FALSE, BatchRenderer.projection_matrix)
    
    glUniform3fv(typeRenderer.shader_locations["lightPos_loc"], 1, light_data_xpos)
    glUniform3fv(typeRenderer.shader_locations["lightDir_loc"], 1, light_data_xdir)
    glUniform3fv(typeRenderer.shader_locations["ambient_loc"], 1, BatchRenderer.light_model["light_ambient"])
    glUniform3fv(typeRenderer.shader_locations["diffuse_loc"], 1, BatchRenderer.light_model["light_diffuse"])
    glUniform3fv(typeRenderer.shader_locations["specular_loc"], 1, BatchRenderer.light_model["light_specular"])
    glUniform1f(typeRenderer.shader_locations["cutoff_loc"], deg2rad(BatchRenderer.light_model["light_cutoff"]))  
    glUniform1f(typeRenderer.shader_locations["exponent_loc"], BatchRenderer.light_model["light_exponent"])  
    glUniform1i(typeRenderer.shader_locations["directional_loc"], BatchRenderer.light_model["light_directional"])  
    glUniform3fv(typeRenderer.shader_locations["attenuation_loc"], 1, BatchRenderer.light_model["light_attenuation"])
    glUniform3fv(typeRenderer.shader_locations["viewPos_loc"], 1, camera_data_pos)
    glUniform3fv(typeRenderer.shader_locations["headlightDir_loc"], 1, forward)
    glUniform1f(typeRenderer.shader_locations["n_env_loc"], BatchRenderer.n_envs) 
    glUniform1f(typeRenderer.shader_locations["res_loc"], BatchRenderer.res)  
    
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
    vao, vbo, normal_vbo, tex_coords_vbo, instance_vbo, ebo, shader_program = setup_opengl(vertices, normals, tex_coords, indices)
    shader_locations = Dict(
        "projection_loc" => glGetUniformLocation(shader_program, "projection"),
        "lightPos_loc" => glGetUniformLocation(shader_program, "lightPos"),
        "lightDir_loc" => glGetUniformLocation(shader_program, "lightDir"),
        "ambient_loc" => glGetUniformLocation(shader_program, "ambient"),
        "diffuse_loc" => glGetUniformLocation(shader_program, "diffuse"),
        "specular_loc" => glGetUniformLocation(shader_program, "specular"),
        "cutoff_loc" => glGetUniformLocation(shader_program, "cutoff"),
        "exponent_loc" => glGetUniformLocation(shader_program, "exponent"),
        "directional_loc" => glGetUniformLocation(shader_program, "directional"),
        "attenuation_loc" => glGetUniformLocation(shader_program, "attenuation"),
        "viewPos_loc" => glGetUniformLocation(shader_program, "viewPos"),
        "headlightDir_loc" => glGetUniformLocation(shader_program, "headlightDir"),
        "view_loc" => glGetUniformLocation(shader_program, "view"),
        "n_env_loc" => glGetUniformLocation(shader_program, "n_env"),
        "res_loc" => glGetUniformLocation(shader_program, "res")
    )

    return GeomRenderer(instance_vbo, shader_program, indices === nothing ? 0 : size(indices ,1), vao, shader_locations)

end
