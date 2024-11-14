struct GeomRenderer
    instance_vbo::UInt32
    #shader_program::Int32
    sizeindices::Int32
    vao::UInt32
    #shader_locations::Dict{String, Int32}
end

function render_geoms(typeRenderer::GeomRenderer, instance_data)
    
    glBindBuffer(GL_ARRAY_BUFFER, typeRenderer.instance_vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(instance_data), instance_data, GL_DYNAMIC_DRAW)
    
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

    return GeomRenderer(instance_vbo[], indices === nothing ? 0 : size(indices ,1), vao[])

end
