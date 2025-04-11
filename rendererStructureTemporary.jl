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



# Load the shared library
const LIBEGL = "/home/utente/Desktop/libegl_example.so"


function setup_egl(width::Cint, height::Cint)
    # `setup_egl` now takes two integer arguments: width and height
    return ccall((:setup_egl, LIBEGL), Cint, (Cint, Cint), width, height)
end

struct GeomRenderer
    instance_vbo::UInt32
    shader_program::Int32
    sizeindices::Int32
    vao::UInt32
    shader_locations::Dict{String, Int32}
end

struct BatchRenderer
    res::Int32
    n_env_cols::Int32      # number of columns in the grid
    n_env_rows::Int32      # number of rows in the grid
    egl_ctx_success::Int32
    projection_matrix::Matrix{Float32}
    light_model::Dict{String, Any}
    texture_array::GLuint                   # the texture array (GPU handle)
    n_geoms_by_type::Dict{String, Int32}
    renderers::Vector{GeomRenderer}
end

function BatchRenderer(model; res, n_env_cols, n_env_rows)
    # Initialise MuJoCo data and scene.
    data = MuJoCo.init_data(model)
    scn = MuJoCo.VisualiserScene()
    cam = MuJoCo.VisualiserCamera()
    opt = MuJoCo.VisualiserOption()
    pert = MuJoCo.VisualiserPerturb()
    MuJoCo.mjv_makeScene(model, scn, 1000)
    catmask = 7
    mjv_updateScene(model, data, opt, pert, cam, catmask, scn)
    n_geoms = scn.ngeom

    # Create EGL context using grid dimensions.
    egl_ctx_success = setup_egl(Cint(n_env_cols * res), Cint(n_env_rows * res))

    # Set up camera projection.
    camera_fovy = extract_camera_model(model)
    projection_matrix = Float32.(collect(transpose(perspective(Float32(deg2rad(camera_fovy)),
                                                              Float32(1.0),
                                                              Float32(0.1),
                                                              Float32(1000.0)))))

    # Extract light parameters.
    light_ambient, light_attenuation, light_cutoff, light_diffuse, light_dir, light_directional, light_exponent, light_pos, light_specular = extract_light_model(model)
    light_model = Dict(
        "light_ambient"     => light_ambient,
        "light_attenuation" => light_attenuation,
        "light_cutoff"      => light_cutoff,
        "light_diffuse"     => light_diffuse,
        "light_dir"         => light_dir,
        "light_directional" => light_directional,
        "light_exponent"    => light_exponent,
        "light_pos"         => light_pos,
        "light_specular"    => light_specular
    )

    # Count geoms by type.
    nboxes, nspheres, nplanes, ncapsules = count_geoms(scn.geoms, n_geoms)
    n_geoms_by_type = Dict(
        "nboxes"    => Int32(nboxes),
        "nspheres"  => Int32(nspheres),
        "nplanes"   => Int32(nplanes),
        "ncapsules" => Int32(ncapsules)
    )


    texture_array = extract_textures(model)

    # Initialise renderers.
    boxRenderer     = BoxRenderer()
    sphereRenderer  = SphereRenderer()
    capsuleRenderer = CapsuleRenderer()
    planeRenderer   = PlaneRenderer()
    renderers = [boxRenderer, sphereRenderer, capsuleRenderer, planeRenderer]

    return BatchRenderer(res, n_env_cols, n_env_rows, egl_ctx_success, projection_matrix,
                         light_model, texture_array, n_geoms_by_type, renderers)
end



function extract_geom_data(model, datas, n_env_cols, n_env_rows, countgeoms)
    total_env = n_env_cols * n_env_rows

    # 22 floats per instance: model(16) + rgba(4) + texLayer(1) + offset(1)
    matrix_instance_data_boxes    = zeros(Float32, 22, total_env * countgeoms["nboxes"])
    matrix_instance_data_spheres  = zeros(Float32, 22, total_env * countgeoms["nspheres"])
    matrix_instance_data_capsules = zeros(Float32, 22, total_env * countgeoms["ncapsules"])
    matrix_instance_data_planes   = zeros(Float32, 22, total_env * countgeoms["nplanes"])
    
    camera_data_pos_all = Vector{Any}(undef, total_env)
    camera_data_mat_all = Vector{Any}(undef, total_env)
    light_data_xdir_all = Vector{Any}(undef, total_env)
    light_data_xpos_all = Vector{Any}(undef, total_env)
    
    for j in 1:total_env
        cam_pos, cam_mat = extract_camera_data(datas[j])
        light_xdir, light_xpos = extract_light_data(datas[j], 1)
        camera_data_pos_all[j] = Float32.(cam_pos)
        camera_data_mat_all[j] = cam_mat
        light_data_xdir_all[j] = light_xdir
        light_data_xpos_all[j] = light_xpos

        offset = Float32(j - 1)

        box_counter     = 0
        sphere_counter  = 0
        capsule_counter = 0
        plane_counter   = 0

        n_geoms::Int32 = model.ngeom

        for i in 1:n_geoms
            geom_type::Int32 = model.geom_type[i]
            pos = SVector{3, Float32}(datas[j].geom_xpos[i, :])
            xmat = SVector{9, Float32}(datas[j].geom_xmat[i, :])
            rgba = SVector{4, Float32}(model.geom_rgba[i, :])
            matid = model.geom_matid[i]

			tex_id::Int32 = (matid >= 0 && matid < length(model.mat_texid)) ? model.mat_texid[matid + 1] : -1


            geomsize = SVector{3, Float32}(model.geom_size[i, :])
            if geom_type == 2
                geomsize = SVector{3, Float32}(geomsize[1], geomsize[1], geomsize[1])
            elseif geom_type == 0
                if geomsize[1] == 0; geomsize = SVector{3, Float32}(5.0, geomsize[2], geomsize[3]); end
                if geomsize[2] == 0; geomsize = SVector{3, Float32}(geomsize[1], 5.0, geomsize[3]); end
                geomsize = SVector{3, Float32}(geomsize[1], geomsize[2], 1.0)
            elseif geom_type == 3
                geomsize = SVector{3, Float32}(geomsize[1], geomsize[1], geomsize[2])
            end

            M = modelMatrix(pos, xmat, geomsize)

            if geom_type == 6  # Box
                idx = (j - 1) * countgeoms["nboxes"] + box_counter + 1
                box_counter += 1
                matrix_instance_data_boxes[1:16, idx]  = M
                matrix_instance_data_boxes[17:20, idx] = rgba
                matrix_instance_data_boxes[21, idx]    = tex_id
                matrix_instance_data_boxes[22, idx]    = offset

            elseif geom_type == 4 || geom_type == 2  # Sphere
                idx = (j - 1) * countgeoms["nspheres"] + sphere_counter + 1
                sphere_counter += 1
                matrix_instance_data_spheres[1:16, idx]  = M
                matrix_instance_data_spheres[17:20, idx] = rgba
                matrix_instance_data_spheres[21, idx]    = tex_id
                matrix_instance_data_spheres[22, idx]    = offset

            elseif geom_type == 0  # Plane
                idx = (j - 1) * countgeoms["nplanes"] + plane_counter + 1
                plane_counter += 1
                matrix_instance_data_planes[1:16, idx]  = M
                if rgba == [0.5, 0.5, 0.5, 1]
                	rgba = [1, 1, 1, 1] #scene would set to [1, 1, 1, 1] default rather than model
                end
                matrix_instance_data_planes[17:20, idx] = rgba
                matrix_instance_data_planes[21, idx]    = tex_id
                matrix_instance_data_planes[22, idx]    = offset

            elseif geom_type == 3  # Capsule
                idx = (j - 1) * countgeoms["ncapsules"] + capsule_counter + 1
                capsule_counter += 1
                matrix_instance_data_capsules[1:16, idx]  = M
                matrix_instance_data_capsules[17:20, idx] = rgba
                matrix_instance_data_capsules[21, idx]    = tex_id
                matrix_instance_data_capsules[22, idx]    = offset
            end
        end
    end

    return matrix_instance_data_boxes, matrix_instance_data_spheres, matrix_instance_data_planes, matrix_instance_data_capsules,
           camera_data_pos_all, camera_data_mat_all, light_data_xdir_all, light_data_xpos_all
end





using Colors, FileIO, Images

function extract_textures(model, target_width::Int=512, target_height::Int=512)
    nt = model.ntex
    println("Number of textures in model: ", nt)
    if nt <= 0
        return 0  # No textures present in model
    end

    # Generate and bind the OpenGL texture array
    texture_array = Ref{GLuint}(0)
    glGenTextures(1, texture_array)
    glBindTexture(GL_TEXTURE_2D_ARRAY, texture_array[])

    # Allocate immutable storage for all layers
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGB8, target_width, target_height, nt)

    # Offset tracker in flat tex_rgb array
    offset = 0

    for i in 0:(nt - 1)
        width = model.tex_width[i + 1]
        height = model.tex_height[i + 1]
        nbytes = width * height * 3

        rgb_slice = model.tex_rgb[offset + 1 : offset + nbytes]
        offset += nbytes

        img = reshape(reinterpret(RGB{N0f8}, rgb_slice), width, height)
        img_resized = imresize(img, (target_height, target_width))  # (H, W)

        save("resized_texture_$(i).png", img_resized)
        @info "Saved resized_texture_$(i).png" original_width=width original_height=height resized_width=target_width resized_height=target_height

        img_bytes = reinterpret(UInt8, img_resized)
        glTexSubImage3D(
            GL_TEXTURE_2D_ARRAY, 0,
            0, 0, i,
            target_width, target_height, 1,
            GL_RGB, GL_UNSIGNED_BYTE, img_bytes
        )
    end


    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glGenerateMipmap(GL_TEXTURE_2D_ARRAY)

    return texture_array[]
end







function render(BatchRenderer, datas)
    instance_data_boxes, instance_data_spheres, instance_data_planes, instance_data_capsules,
    camera_data_pos_all, camera_data_mat_all, light_data_xdir_all, light_data_xpos_all =
        extract_geom_data(model, datas, BatchRenderer.n_env_cols, BatchRenderer.n_env_rows, BatchRenderer.n_geoms_by_type)

    total_env = BatchRenderer.n_env_cols * BatchRenderer.n_env_rows
    glEnable(GL_DEPTH_TEST)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    nboxes    = BatchRenderer.n_geoms_by_type["nboxes"]
    nspheres  = BatchRenderer.n_geoms_by_type["nspheres"]
    nplanes   = BatchRenderer.n_geoms_by_type["nplanes"]
    ncapsules = BatchRenderer.n_geoms_by_type["ncapsules"]

    for j in 1:total_env
        offset_boxes    = ((j-1)*nboxes + 1):(j*nboxes)
        offset_spheres  = ((j-1)*nspheres + 1):(j*nspheres)
        offset_capsules = ((j-1)*ncapsules + 1):(j*ncapsules)
        offset_planes   = ((j-1)*nplanes + 1):(j*nplanes)

        boxes_subset    = instance_data_boxes[:, offset_boxes]
        spheres_subset  = instance_data_spheres[:, offset_spheres]
        capsules_subset = instance_data_capsules[:, offset_capsules]
        planes_subset   = instance_data_planes[:, offset_planes]

        cam_pos = camera_data_pos_all[j]
        cam_mat = camera_data_mat_all[j]
        light_dir = light_data_xdir_all[j]
        light_pos = light_data_xpos_all[j]

        display(BatchRenderer.renderers[1], boxes_subset,    light_dir, light_pos, cam_mat, cam_pos, BatchRenderer)
        display(BatchRenderer.renderers[2], spheres_subset,  light_dir, light_pos, cam_mat, cam_pos, BatchRenderer)
        display(BatchRenderer.renderers[3], capsules_subset, light_dir, light_pos, cam_mat, cam_pos, BatchRenderer)
        display(BatchRenderer.renderers[4], planes_subset,   light_dir, light_pos, cam_mat, cam_pos, BatchRenderer)
    end

    return save_egl_image("rendered_image_julia.png", BatchRenderer.n_env_cols * BatchRenderer.res, BatchRenderer.n_env_rows * BatchRenderer.res)
end



function display(typeRenderer::GeomRenderer, instance_data, light_data_xdir, light_data_xpos, camera_data_mat, camera_data_pos, BatchRenderer::BatchRenderer)
    final, forward = lookatmatrix(camera_data_pos, camera_data_mat)

    # Upload the per-instance data including the new offset
    setup_data(instance_data, typeRenderer.instance_vbo)

    glUseProgram(typeRenderer.shader_program)

    # View and projection
    glUniformMatrix4fv(typeRenderer.shader_locations["view_loc"], 1, GL_FALSE, final)
    glUniformMatrix4fv(typeRenderer.shader_locations["projection_loc"], 1, GL_FALSE, BatchRenderer.projection_matrix)

    # Lighting uniforms
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

    # Grid layout
    glUniform1f(typeRenderer.shader_locations["n_env_cols_loc"], BatchRenderer.n_env_cols)
    glUniform1f(typeRenderer.shader_locations["n_env_rows_loc"], BatchRenderer.n_env_rows)
    glUniform1f(typeRenderer.shader_locations["res_loc"], BatchRenderer.res)

    # Bind the texture array
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D_ARRAY, BatchRenderer.texture_array)
    glUniform1i(typeRenderer.shader_locations["texture_array_loc"], 0)

    glBindVertexArray(typeRenderer.vao)

    n_instances = size(instance_data, 2)
    if typeRenderer.sizeindices == 0
        glDrawArraysInstanced(GL_TRIANGLES, 0, 36, n_instances)
    else
        glDrawElementsInstanced(GL_TRIANGLES, typeRenderer.sizeindices, GL_UNSIGNED_INT, C_NULL, n_instances)
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
    vao, vbo, normal_vbo, tex_coord_vbo, instance_vbo, ebo, shader_program = setup_opengl(vertices, normals, tex_coords, indices)
    shader_locations = Dict(
        "projection_loc"    => glGetUniformLocation(shader_program, "projection"),
        "lightPos_loc"      => glGetUniformLocation(shader_program, "lightPos"),
        "lightDir_loc"      => glGetUniformLocation(shader_program, "lightDir"),
        "ambient_loc"       => glGetUniformLocation(shader_program, "ambient"),
        "diffuse_loc"       => glGetUniformLocation(shader_program, "diffuse"),
        "specular_loc"      => glGetUniformLocation(shader_program, "specular"),
        "cutoff_loc"        => glGetUniformLocation(shader_program, "cutoff"),
        "exponent_loc"      => glGetUniformLocation(shader_program, "exponent"),
        "directional_loc"   => glGetUniformLocation(shader_program, "directional"),
        "attenuation_loc"   => glGetUniformLocation(shader_program, "attenuation"),
        "viewPos_loc"       => glGetUniformLocation(shader_program, "viewPos"),
        "headlightDir_loc"  => glGetUniformLocation(shader_program, "headlightDir"),
        "view_loc"          => glGetUniformLocation(shader_program, "view"),
        # Uniforms for grid layout:
        "n_env_cols_loc"    => glGetUniformLocation(shader_program, "n_env_cols"),
        "n_env_rows_loc"    => glGetUniformLocation(shader_program, "n_env_rows"),
        "res_loc"           => glGetUniformLocation(shader_program, "res"),
        # New uniform for the texture array.
        "texture_array_loc" => glGetUniformLocation(shader_program, "texture_array")
    )

    return GeomRenderer(instance_vbo, shader_program, indices === nothing ? 0 : size(indices, 1), vao, shader_locations)
end




    



#function display(geomRenderer::GeomRenderer)
    #Same for all geom renderers
#end

#function extract(...)

#    const PLANE = 0
#    const SPHERE = 
#    const BOX = 
#    const ELLIPSOID =
#    const CAPSULE =


#    buffers = Dict{Int64, Matrix{Float32}}()
#    buffers[SPHERE] = zeros(Float32, n_spheres*n_envs, 22)

#    for env=...
#        for obj in scene
#            if obj.type not in buffers.keys()
#                continue
#            end
#            buffers[obj.type][ind...] = ... 
#        end        
#    end
#    return buffers
#end

function count_geoms(scene_geoms::Ptr{MuJoCo.LibMuJoCo.mjvGeom_}, num_geoms::Int32)
    # Initialize counts for each type
    box_count = 0
    sphere_count = 0
    plane_count = 0
    capsule_count = 0

    # Loop through each geom and count its type
    for i in 0:num_geoms-1
        # Compute the pointer to the current geom
        geom_ptr = Ptr{MuJoCo.LibMuJoCo.mjvGeom_}(scene_geoms + i * sizeof(MuJoCo.LibMuJoCo.mjvGeom_))
        
        # Load the geom structure
        geom = unsafe_load(geom_ptr)   
        geom_type = geom.type
        
        # Count based on the geom type
        if geom_type == 6
            box_count += 1
        elseif geom_type == 4 || geom_type == 2
            sphere_count += 1
        elseif geom_type == 0
            plane_count += 1
        elseif geom_type == 3
            capsule_count += 1
        end
    end
    
    return box_count, sphere_count, plane_count, capsule_count
end


using StaticArrays

using StaticArrays

function modelMatrix(pos::SVector{3, Float32}, mat::SVector{9, Float32}, geomsize::SVector{3, Float32})
    # Extract scaling components
    sx, sy, sz = geomsize

    # Extract translation components
    tx, ty, tz = pos

    # Extract rotation matrix components (assuming mat is a flat tuple)
    r00, r01, r02 = mat[1:3]
    r10, r11, r12 = mat[4:6]
    r20, r21, r22 = mat[7:9]

    # Directly initialize the static vector M with the transformation matrix
    M = SVector{16, Float32}(
        r00 * sx, r10 * sx, r20 * sx, 0.0f0,
        r01 * sy, r11 * sy, r21 * sy, 0.0f0,
        r02 * sz, r12 * sz, r22 * sz, 0.0f0,
        tx, ty, tz, 1.0f0
    )

    return M
end



function modelMatrix_transposed!(M::Vector{Float32}, pos::Vector{Float32}, mat::Vector{Float32}, geomsize::Vector{Float32})
    # Extract scaling components
    sx, sy, sz = geomsize

    # Extract translation components
    tx, ty, tz = pos

    # Extract rotation matrix components (assuming mat is a flat tuple)
    r00, r01, r02 = mat[1:3]
    r10, r11, r12 = mat[4:6]
    r20, r21, r22 = mat[7:9]
    
    # Fill in the transposed transformation matrix M directly
    M[1]  = r00 * sx
    M[2]  = r01 * sy
    M[3]  = r02 * sz
    M[4]  = tx
    
    M[5]  = r10 * sx
    M[6]  = r11 * sy
    M[7]  = r12 * sz
    M[8]  = ty
    
    M[9]  = r20 * sx
    M[10] = r21 * sy
    M[11] = r22 * sz
    M[12] = tz

    M[13] = 0.0f0
    M[14] = 0.0f0
    M[15] = 0.0f0
    M[16] = 1.0f0
end




function extract_camera_data(data)
	#extract side (second camera)
    # Iterate over each row in data.cam_xpos
    for i in 1:size(data.cam_xpos, 1)
    	#fixed at second camera (side for rodent)
        camera_position = data.cam_xpos[2, :]
        camera_mat = Float32.(transpose(reshape(data.cam_xmat[2, :], 3, 3)))
        #can add more if needed, but just position and mat used in python code
        return camera_position, camera_mat
    end
end


function perspective(fovy::Float32, aspect::Float32, near::Float32, far::Float32)
    f = 1.0 / tan(fovy / 2.0)
    nf = 1.0 / (near - far)

    # Create the 4x4 perspective projection matrix
    return Matrix{Float32}([
        f / aspect  0.0  0.0                          0.0
        0.0         f    0.0                          0.0
        0.0         0.0  (far + near) * nf            -1.0
        0.0         0.0  (2.0 * far * near) * nf      0.0
    ])

end


function extract_light_model(model)
    # Create arrays to hold extracted light data
    light_active = []
    light_ambient = []
    light_attenuation = []
    light_bodyid = []
    light_bulbradius = []
    light_castshadow = []
    light_cutoff = []
    light_diffuse = []
    light_dir = []
    light_dir0 = []
    light_directional = []
    light_exponent = []
    light_mode = []
    light_pos = []
    light_pos0 = []
    light_poscom0 = []
    light_specular = []
    light_targetbodyid = []

    # Iterate over each light in the model
    for i in 1:model.nlight
        push!(light_active, model.light_active[i])
        push!(light_ambient, model.light_ambient[i, :])
        push!(light_attenuation, model.light_attenuation[i, :])
        push!(light_bodyid, model.light_bodyid[i])
        push!(light_bulbradius, model.light_bulbradius[i])
        push!(light_castshadow, model.light_castshadow[i])
        push!(light_cutoff, model.light_cutoff[i])
        push!(light_diffuse, model.light_diffuse[i, :])
        push!(light_dir, model.light_dir[i, :])
        push!(light_dir0, model.light_dir0[i, :])
        push!(light_directional, model.light_directional[i])
        push!(light_exponent, model.light_exponent[i])
        push!(light_mode, model.light_mode[i])
        push!(light_pos, model.light_pos[i, :])
        push!(light_pos0, model.light_pos0[i, :])
        push!(light_poscom0, model.light_poscom0[i, :])
        push!(light_specular, model.light_specular[i, :])
        push!(light_targetbodyid, model.light_targetbodyid[i])
    end

    return light_ambient[1], light_attenuation[1], light_cutoff[1], light_diffuse[1], light_dir[1], Int32.(light_directional[1]), light_exponent[1], Float32.(light_pos[1]), light_specular[1]
end


function extract_light_data(data, index::Int)
    # Extract the light direction and position for the given index
    light_xdir = Float32.(data.light_xdir[index, :])
    light_xpos = Float32.(data.light_xpos[index, :])
    return light_xdir, light_xpos
end



function lookAt(eye::Vector{Float32}, center::Vector{Float32}, up::Vector{Float32})
    # Forward vector (normalized)
    f = normalize(center - eye)
    
    # Right vector (normalized)
    s = normalize(cross(f, up))
    
    # Up vector
    u = cross(s, f)

    # Create the 4x4 view matrix
    view_matrix = Matrix{Float32}(I, 4, 4)
    
    # Assign vectors to the matrix rows (transposed to handle the row assignment)
    view_matrix[1, 1:3] = s'
    view_matrix[2, 1:3] = u'
    view_matrix[3, 1:3] = -f'

    # Translation part
    view_matrix[1, 4] = -dot(s, eye)
    view_matrix[2, 4] = -dot(u, eye)
    view_matrix[3, 4] = dot(f, eye)
    
    return view_matrix
end

function lookatmatrix(position::Vector{Float32}, mat::Matrix{Float32})

    # Extract the right (X), up (Y), and forward (Z) vectors from the orientation matrix
    forward = -mat[:, 3]  # Forward is the negative Z direction
    right = mat[:, 1]     # Right is the X direction
    up = mat[:, 2]        # Up is the Y direction

    # Compute the lookAt matrix: eye (position), center (position + forward), and up vector
    final = lookAt(position, position + forward, up)
    
    return final, forward
end



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



function extract_camera_model(model)
    # Iterate over each row in model.cam_pos
    for i in 1:size(model.cam_pos, 1)
        camera_position = model.cam_pos[i, :]
        camera_fovy = model.cam_fovy[i]
        #can add more if needed, but just fovy used in python code
        return camera_fovy
    end
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
    normal = cross(v0, v1)
    normal = normal / norm(normal)  # Normalize the normal vector

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

    # Vertex data
    glBindBuffer(GL_ARRAY_BUFFER, vbo[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), C_NULL)
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, normal_vbo[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(normals), normals, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), C_NULL)
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, tex_coord_vbo[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(tex_coords), tex_coords, GL_STATIC_DRAW)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), C_NULL)
    glEnableVertexAttribArray(2)

    # Optional index buffer
    if indices !== nothing
        ebo = Ref(GLuint(0))
        glGenBuffers(1, ebo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo[])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW)
    else
        ebo = nothing
    end

    # Instance attributes: 22 floats total
    # modelMatrix: 0–15 (locations 3-6)
    # rgba: 16–19 (location 7)
    # texLayer: 20 (location 8)
    # offset: 21 (location 9)
    glBindBuffer(GL_ARRAY_BUFFER, instance_vbo[])
    instance_stride = 22 * sizeof(GL_FLOAT)

    for i in 0:3
        glVertexAttribPointer(3 + i, 4, GL_FLOAT, GL_FALSE, instance_stride,
                              Ptr{Cvoid}(i * 4 * sizeof(GLfloat)))
        glEnableVertexAttribArray(3 + i)
        glVertexAttribDivisor(3 + i, 1)
    end

    glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, instance_stride,
                          Ptr{Cvoid}(16 * sizeof(GLfloat)))
    glEnableVertexAttribArray(7)
    glVertexAttribDivisor(7, 1)

    glVertexAttribPointer(8, 1, GL_FLOAT, GL_FALSE, instance_stride,
                          Ptr{Cvoid}(20 * sizeof(GLfloat)))
    glEnableVertexAttribArray(8)
    glVertexAttribDivisor(8, 1)

    glVertexAttribPointer(9, 1, GL_FLOAT, GL_FALSE, instance_stride,
                          Ptr{Cvoid}(21 * sizeof(GLfloat)))
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
    glBindBuffer(GL_ARRAY_BUFFER, instance_vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(instance_data), instance_data, GL_DYNAMIC_DRAW)
end


function compile_shaders()
    vertex_source = Vector{UInt8}(codeunits("""
        #version 330 core

        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aNormal;
        layout(location = 2) in vec2 aTexCoords;
        layout(location = 3) in mat4 modelMatrix;
        layout(location = 7) in vec4 rgba;
        layout(location = 8) in float texLayer;
        layout(location = 9) in float offset;

        out vec4 vColor;
        out vec3 Normal;
        out vec3 FragPos;
        out vec2 texCoords;
        out float textureIndex;
        out vec2 frag_bounds_min;
        out vec2 frag_bounds_max;

        uniform mat4 view;
        uniform mat4 projection;
        uniform float n_env_cols;
        uniform float n_env_rows;
        uniform float res;

        void main()
        {
            float offset_x = mod(offset, n_env_cols);
            float offset_y = floor(offset / n_env_cols);

            float scale_x = 1.0 / n_env_cols;
            float scale_y = 1.0 / n_env_rows;

            float trans_x = -1.0 + (2.0 * offset_x + 1.0) * scale_x;
            float trans_y = -1.0 + (2.0 * offset_y + 1.0) * scale_y;

            mat4 envScale = mat4(
                scale_x, 0.0,     0.0, 0.0,
                0.0,     scale_y, 0.0, 0.0,
                0.0,     0.0,     1.0, 0.0,
                0.0,     0.0,     0.0, 1.0
            );

            mat4 envTranslate = mat4(
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                trans_x, trans_y, 0.0, 1.0
            );

            vec4 worldPos = modelMatrix * vec4(aPos, 1.0);
            gl_Position = envTranslate * envScale * projection * view * worldPos;

            vColor = rgba;
            Normal = normalize(mat3(transpose(inverse(modelMatrix))) * aNormal);
            FragPos = vec3(envTranslate * envScale * view * worldPos);

            vec4 texCoordH = modelMatrix * vec4(aTexCoords, 0.0, 1.0);
            texCoords = texCoordH.xy / texCoordH.w;

            textureIndex = texLayer;

            // Calculate pixel bounds for current environment
            frag_bounds_min = vec2(offset_x * res, offset_y * res);
            frag_bounds_max = vec2((offset_x + 1.0) * res, (offset_y + 1.0) * res);
        }
    """))

    fragment_source = Vector{UInt8}(codeunits("""
		#version 330 core

		in vec4 vColor;              // Color passed from vertex shader (RGBA)
		in vec3 Normal;              // Normal passed from vertex shader
		in vec3 FragPos;             // Fragment position in world space
		in vec2 texCoords;          // Texture coordinates passed from vertex shader
		in float textureIndex;      // Index of the texture layer (or -1 when no texture)
		in vec2 frag_bounds_min;    // Minimum bounds of the fragment
		in vec2 frag_bounds_max;    // Maximum bounds of the fragment

		out vec4 FragColor;

		uniform vec3 lightPos;
		uniform vec3 lightDir;
		uniform vec3 ambient;
		uniform vec3 diffuse;
		uniform vec3 specular;
		uniform float cutoff;
		uniform float exponent;
		uniform int directional;
		uniform vec3 attenuation;
		uniform vec3 viewPos;
		uniform vec3 headlightDir;

		uniform sampler2DArray texture_array;

		void main()
		{
			// Discard fragments outside their environment viewport
			vec2 frag_coord = gl_FragCoord.xy;
			if (frag_coord.x < frag_bounds_min.x || frag_coord.x > frag_bounds_max.x ||
				frag_coord.y < frag_bounds_min.y || frag_coord.y > frag_bounds_max.y)
			{
				discard;
			}

			// Get texture color as base color (albedo) if textureIndex is valid, else use RGBA color (vColor)
			vec3 albedo;
			if (textureIndex >= 0.0) {
				albedo = texture(texture_array, vec3(texCoords, textureIndex)).rgb;
			} else {
				albedo = vColor.rgb;  // Use geometry's RGBA color when no texture
			}

			vec3 norm = normalize(Normal);
			vec3 result = ambient * albedo;

			// Point light (when directional == 0)
			if (directional == 0)
			{
				vec3 lightDirection = normalize(lightPos - FragPos);
				float distance = length(lightPos - FragPos);
				float attenuationFactor = 1.0 / (attenuation.x + attenuation.y * distance + attenuation.z * distance * distance);

				float diff = max(dot(norm, lightDirection), 0.0);
				result += diff * diffuse * albedo * attenuationFactor;

				vec3 viewDir = normalize(viewPos - FragPos);
				vec3 reflectDir = reflect(-lightDirection, norm);
				float specVal = pow(max(dot(viewDir, reflectDir), 0.0), exponent);
				result += specVal * specular * albedo * attenuationFactor;
			}
			else  // Directional light
			{
				vec3 dir = normalize(-lightDir);
				float diff = max(dot(norm, dir), 0.0);
				result += diff * diffuse * albedo;

				vec3 viewDir = normalize(viewPos - FragPos);
				vec3 reflectDir = reflect(-dir, norm);
				float specVal = pow(max(dot(viewDir, reflectDir), 0.0), exponent);
				result += specVal * specular * albedo;
			}

			// Headlight effect
			vec3 hDir = normalize(-headlightDir);
			result += 0.1 * albedo; // ambient
			result += max(dot(norm, hDir), 0.0) * 0.4 * albedo; // diffuse
			vec3 viewHDir = normalize(viewPos - FragPos);
			vec3 reflectHDir = reflect(-hDir, norm);
			float specH = pow(max(dot(viewHDir, reflectHDir), 0.0), exponent);
			result += specH * 0.5 * albedo; // specular

			FragColor = vec4(result, vColor.a);  // Optional: modulate by alpha
		}


    """))

    # Compile and link shaders
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, 1, Ptr{UInt8}[pointer(vertex_source)], Ref{GLint}(length(vertex_source)))
    glCompileShader(vertex_shader)
    status = Ref(GLint(0))
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, status)
    if status[] != GL_TRUE
        buffer = Array(UInt8, 512)
        glGetShaderInfoLog(vertex_shader, 512, C_NULL, buffer)
        @error "$(unsafe_string(pointer(buffer), 512))"
    end

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, 1, Ptr{UInt8}[pointer(fragment_source)], Ref{GLint}(length(fragment_source)))
    glCompileShader(fragment_shader)
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, status)
    if status[] != GL_TRUE
        buffer = Array(UInt8, 512)
        glGetShaderInfoLog(fragment_shader, 512, C_NULL, buffer)
        @error "$(unsafe_string(pointer(buffer), 512))"
    end

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    glUseProgram(shader_program)

    return shader_program
end




function save_egl_image(filename::String, width::Int32, height::Int32)
    glReadBuffer(GL_FRONT)

    pixels = Vector{UInt8}(undef, width * height * 3)

    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels)

	image_data = reshape(pixels, (3, width, height))

	image_data = reverse(image_data, dims=3)

    # Convert to Float32 and normalize values to [0.0, 1.0]
    float_image_data = convert(Array{Float32, 3}, image_data) / 255.0

    # Create an image object using RGB with Float32 values
    img = collect(colorview(RGB, float_image_data)')

    # Save the image as a PNG file
    save(filename, img)
    #return img
    
end







