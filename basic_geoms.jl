function generate_box_vertices()
    vertices = collect(transpose(Float32[
        # positions
        -1.0,  1.0,  1.0,  # Front face
        -1.0, -1.0,  1.0,
        1.0, -1.0,  1.0,
        -1.0,  1.0,  1.0,
        1.0, -1.0,  1.0,
        1.0,  1.0,  1.0,
        # Back face
        -1.0,  1.0, -1.0,
        1.0,  1.0, -1.0,
        1.0, -1.0, -1.0,
        -1.0,  1.0, -1.0,
        1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0,
        # Left face
        -1.0,  1.0, -1.0,
        -1.0,  1.0,  1.0,
        -1.0, -1.0,  1.0,
        -1.0,  1.0, -1.0,
        -1.0, -1.0,  1.0,
        -1.0, -1.0, -1.0,
        # Right face
        1.0,  1.0, -1.0,
        1.0,  1.0,  1.0,
        1.0, -1.0,  1.0,
        1.0,  1.0, -1.0,
        1.0, -1.0,  1.0,
        1.0, -1.0, -1.0,
        # Top face
        -1.0,  1.0, -1.0,
        1.0,  1.0, -1.0,
        1.0,  1.0,  1.0,
        -1.0,  1.0, -1.0,
        1.0,  1.0,  1.0,
        -1.0,  1.0,  1.0,
        # Bottom face
        -1.0, -1.0, -1.0,
        -1.0, -1.0,  1.0,
        1.0, -1.0,  1.0,
        -1.0, -1.0, -1.0,
        1.0, -1.0,  1.0,
        1.0, -1.0, -1.0
    ]))

    normals = collect(transpose(Float32[
        # Front face
        0.0 0.0 1.0;  # Normal for front face
        0.0 0.0 1.0;
        0.0 0.0 1.0;
        0.0 0.0 1.0;
        0.0 0.0 1.0;
        0.0 0.0 1.0;
        # Back face
        0.0 0.0 -1.0;  # Normal for back face
        0.0 0.0 -1.0;
        0.0 0.0 -1.0;
        0.0 0.0 -1.0;
        0.0 0.0 -1.0;
        0.0 0.0 -1.0;
        # Left face
        -1.0 0.0 0.0;  # Normal for left face
        -1.0 0.0 0.0;
        -1.0 0.0 0.0;
        -1.0 0.0 0.0;
        -1.0 0.0 0.0;
        -1.0 0.0 0.0;
        # Right face
        1.0 0.0 0.0;  # Normal for right face
        1.0 0.0 0.0;
        1.0 0.0 0.0;
        1.0 0.0 0.0;
        1.0 0.0 0.0;
        1.0 0.0 0.0;
        # Top face
        0.0 1.0 0.0;  # Normal for top face
        0.0 1.0 0.0;
        0.0 1.0 0.0;
        0.0 1.0 0.0;
        0.0 1.0 0.0;
        0.0 1.0 0.0;
        # Bottom face
        0.0 -1.0 0.0;  # Normal for bottom face
        0.0 -1.0 0.0;
        0.0 -1.0 0.0;
        0.0 -1.0 0.0;
        0.0 -1.0 0.0;
        0.0 -1.0 0.0
    ]))

    tex_coords = Float32[
        # Front face
        0.0, 1.0,
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
        # Back face
        1.0, 1.0,
        0.0, 1.0,
        0.0, 0.0,
        1.0, 1.0,
        0.0, 0.0,
        1.0, 0.0,
        # Left face
        1.0, 1.0,
        0.0, 1.0,
        0.0, 0.0,
        1.0, 1.0,
        0.0, 0.0,
        1.0, 0.0,
        # Right face
        1.0, 1.0,
        0.0, 1.0,
        0.0, 0.0,
        1.0, 1.0,
        0.0, 0.0,
        1.0, 0.0,
        # Top face
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
        # Bottom face
        0.0, 1.0,
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    ]

    indices = nothing

    return vertices, indices, normals, tex_coords
end

function generate_sphere_vertices(radius, sector_count=32, stack_count=32)
    vertices = Float32[]
    normals = Float32[]
    texture_coords = Float32[]
    indices = Int32[]

    for i in 0:stack_count
        stack_angle = π / 2 - i * π / stack_count  # from pi/2 to -pi/2
        xy = radius * cos(stack_angle)  # radius * cos(u)
        z = radius * sin(stack_angle)   # radius * sin(u)

        for j in 0:sector_count
            sector_angle = j * 2 * π / sector_count  # from 0 to 2pi

            # Vertex position (x, y, z)
            x = xy * cos(sector_angle)
            y = xy * sin(sector_angle)
            push!(vertices, x, y, z)

            # Normalized normal vector (x, y, z)
            nx = x / radius
            ny = y / radius
            nz = z / radius
            push!(normals, nx, ny, nz)

            # Texture coordinates (s, t)
            s = j / sector_count
            t = i / stack_count
            push!(texture_coords, s, t)
        end
    end

    # Generate indices for the sphere mesh
    for i in 0:(stack_count - 1)
        k1 = i * (sector_count + 1)
        k2 = k1 + sector_count + 1
        for j in 0:(sector_count - 1)
            if i != 0
                push!(indices, k1, k2, k1 + 1)
            end
            if i != (stack_count - 1)
                push!(indices, k1 + 1, k2, k2 + 1)
            end
            k1 += 1
            k2 += 1
        end
    end

    return vertices, indices, normals, texture_coords
end

function generate_capsule_vertices(radius, half_height, sector_count=36, stack_count=18)
    vertices = Float32[]
    indices = Int32[]
    normals = Float32[]
    texture_coords = Float32[]

    function compute_normal(x, y, z)
        length = sqrt(x * x + y * y + z * z)
        return x / length, y / length, z / length
    end

    # Generate cylinder part
    for i in 0:sector_count
        theta = i * 2 * π / sector_count
        x = radius * cos(theta)
        y = radius * sin(theta)

        # Bottom
        push!(vertices, x, y, -half_height / 2)
        nx, ny, nz = compute_normal(x, y, -half_height / 2)
        push!(normals, nx, ny, nz)
        push!(texture_coords, i / sector_count, 0.0)

        # Top
        push!(vertices, x, y, half_height / 2)
        nx, ny, nz = compute_normal(x, y, half_height / 2)
        push!(normals, nx, ny, nz)
        push!(texture_coords, i / sector_count, 1.0)
    end

    # Bottom hemisphere
    for i in 0:stack_count
        phi = π / 2 - i * π / stack_count
        z = radius * sin(phi) - half_height / 2
        xy = radius * cos(phi)

        for j in 0:sector_count
            theta = j * 2 * π / sector_count
            x = xy * cos(theta)
            y = xy * sin(theta)
            push!(vertices, x, y, z)
            nx, ny, nz = compute_normal(x, y, z)
            push!(normals, nx, ny, nz)
            push!(texture_coords, j / sector_count, i / stack_count)
        end
    end

    # Top hemisphere
    for i in 0:stack_count
        phi = π / 2 - i * π / stack_count
        z = radius * sin(phi) + half_height / 2
        xy = radius * cos(phi)

        for j in 0:sector_count
            theta = j * 2 * π / sector_count
            x = xy * cos(theta)
            y = xy * sin(theta)
            push!(vertices, x, y, z)
            nx, ny, nz = compute_normal(x, y, z)
            push!(normals, nx, ny, nz)
            push!(texture_coords, j / sector_count, i / stack_count)
        end
    end

    # Indices for the cylinder
    for i in 0:(sector_count - 1)
        push!(indices, 2 * i, 2 * i + 1, 2 * i + 2)
        push!(indices, 2 * i + 1, 2 * i + 2, 2 * i + 3)
    end

    base_idx = 2 * (sector_count + 1)
    # Indices for the bottom hemisphere
    for i in 0:(stack_count - 1)
        for j in 0:(sector_count - 1)
            k1 = base_idx + i * (sector_count + 1) + j
            k2 = k1 + (sector_count + 1)
            push!(indices, k1, k2, k1 + 1)
            push!(indices, k2, k1 + 1, k2 + 1)
        end
    end

    base_idx += (stack_count + 1) * (sector_count + 1)
    # Indices for the top hemisphere
    for i in 0:(stack_count - 1)
        for j in 0:(sector_count - 1)
            k1 = base_idx + i * (sector_count + 1) + j
            k2 = k1 + (sector_count + 1)
            push!(indices, k1, k2, k1 + 1)
            push!(indices, k2, k1 + 1, k2 + 1)
        end
    end

    return vertices, indices, normals, texture_coords
end

function generate_plane_vertices(half_x::Float32, half_y::Float32)
    # Define the vertices of the plane (Array of Float32s)
    vertices = Float32[
        -half_x  -half_y  0.0f0;  # Bottom-left
         half_x  -half_y  0.0f0;  # Bottom-right
         half_x   half_y  0.0f0;  # Top-right
        -half_x   half_y  0.0f0   # Top-left
    ]

    # Calculate the normal vector for the plane
    v0 = vertices[2, :] .- vertices[1, :]
    v1 = vertices[3, :] .- vertices[1, :]
    normal = LinearAlgebra.cross(v0, v1)
    normal = normal / LinearAlgebra.norm(normal)  # Normalize the normal vector

    # Define the normals for the plane (same for all vertices)
    normals = repeat(normal', 4, 1)

    # Define the texture coordinates for the plane (Array of Float32s)
    texture_coords = Float32[
        0.0f0  0.0f0;  # Bottom-left
        1.0f0  0.0f0;  # Bottom-right
        1.0f0  1.0f0;  # Top-right
        0.0f0  1.0f0   # Top-left
    ]


    indices = UInt32[0, 1, 2,  0, 2, 3] 

    vertices = collect(vertices')
    normals = collect(normals')


    return vertices, indices, normals, texture_coords
end

