function count_geoms(scene)
    counts = DefaultDict{MuJoCo.mjtGeom, Int32}(Int32(0))
    scene_geoms = scene.geoms
    for i in 0:(scene.ngeom - 1)
        # Compute the pointer to the current geom
        geom_ptr = Ptr{MuJoCo.LibMuJoCo.mjvGeom_}(scene_geoms + i * sizeof(MuJoCo.LibMuJoCo.mjvGeom_))
        
        # Load the geom structure
        geom = unsafe_load(geom_ptr)
        geom_type = MuJoCo.mjtGeom(geom.type)
        counts[geom_type] += 1
    end
    
    #Count ellipsoids as spheres
    counts[MuJoCo.mjGEOM_SPHERE] += counts[MuJoCo.mjGEOM_ELLIPSOID]
    counts[MuJoCo.mjGEOM_ELLIPSOID] = 0

    return counts
end

function extract_geom_data(batchRenderer, datas)
    model = batchRenderer.model
    geom_counts = batchRenderer.geom_counts
    n_env = length(datas)
    supported_geoms = [MuJoCo.mjGEOM_PLANE, MuJoCo.mjGEOM_SPHERE,
                       MuJoCo.mjGEOM_CAPSULE, MuJoCo.mjGEOM_BOX]
    matrix_instance_data = Dict(geom_id => zeros(Float32, 22, n_env * geom_counts[geom_id])
                                for geom_id in supported_geoms)
    for j in 1:n_env
        counters = Dict(geom_id => 0 for geom_id in supported_geoms)
        n_geoms::Int32 = model.ngeom

        # Loop over geometries
        for i in 1:n_geoms
            geom_type = MuJoCo.mjtGeom(model.geom_type[i])

			pos = SVector{3, Float32}(view(datas[j].geom_xpos, i, :))
            xmat = SVector{9, Float32}(view(datas[j].geom_xmat, i, :))
            rgba = SVector{4, Float32}(view(model.geom_rgba, i, :))

            offset::Int32 = j - 1

            #tex_id = Float32(geom.texid)
            #if tex_id != -1
                #tex_id = get(geomTexID_to_openglTexID, tex_id, -1)
                #tex_id = Float32(-1)
            #end
            tex_id::Int32 = -1

            # Initialize the geomsize array
            geomsize = SVector{3, Float32}(model.geom_size[i, :])

            if geom_type == MuJoCo.mjGEOM_SPHERE
				geomsize = SVector{3, Float32}(geomsize[1], geomsize[1], geomsize[1])
            elseif geom_type == MuJoCo.mjGEOM_PLANE
                if geomsize[1] == 0
                    geomsize = SVector{3, Float32}(5.0, geomsize[2], geomsize[3])
                end
                if geomsize[2] == 0
                    geomsize = SVector{3, Float32}(geomsize[1], 5.0, geomsize[3])
                end
				geomsize = SVector{3, Float32}(geomsize[1], geomsize[2], 1.0)
            elseif geom_type == MuJoCo.mjGEOM_CAPSULE
                # for scene geomsize = Float32[geomsize[2], geomsize[2], geomsize[3]]
                geomsize = SVector{3, Float32}(geomsize[1], geomsize[1], geomsize[2]) #for model
            end

            # Compute the model matrix, M is pre-allocated and reused
            M = modelMatrix(pos, xmat, geomsize)
            if geom_type == MuJoCo.mjGEOM_ELLIPSOID
                geom_type = MuJoCo.mjGEOM_SPHERE
            end
            geom_idx = (j - 1) * batchRenderer.geom_counts[geom_type] + counters[geom_type] + 1
			matrix_instance_data[geom_type][1:16,  geom_idx] = M
			matrix_instance_data[geom_type][17:20, geom_idx] = rgba
			matrix_instance_data[geom_type][21,    geom_idx] = offset
			matrix_instance_data[geom_type][22,    geom_idx] = tex_id
        end
    end    

    return matrix_instance_data
end

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
    light_model = DefaultDict(Vector) #TODO: Should set a eltype for the default array

    # Iterate over each light in the model
    for i in 1:model.nlight
        push!(light_model["light_active"], model.light_active[i])
        push!(light_model["light_ambient"], model.light_ambient[i, :])
        push!(light_model["light_attenuation"], model.light_attenuation[i, :])
        push!(light_model["light_bodyid"], model.light_bodyid[i])
        push!(light_model["light_bulbradius"], model.light_bulbradius[i])
        push!(light_model["light_castshadow"], model.light_castshadow[i])
        push!(light_model["light_cutoff"], model.light_cutoff[i])
        push!(light_model["light_diffuse"], model.light_diffuse[i, :])
        push!(light_model["light_dir"], model.light_dir[i, :])
        push!(light_model["light_dir0"], model.light_dir0[i, :])
        push!(light_model["light_directional"], Int32.(model.light_directional[i]))
        push!(light_model["light_exponent"], model.light_exponent[i])
        push!(light_model["light_mode"], model.light_mode[i])
        push!(light_model["light_pos"], Float32.(model.light_pos[i, :]))
        push!(light_model["light_pos0"], model.light_pos0[i, :])
        push!(light_model["light_poscom0"], model.light_poscom0[i, :])
        push!(light_model["light_specular"], model.light_specular[i, :])
        push!(light_model["light_targetbodyid"], model.light_targetbodyid[i])
    end
    light_model = Dict(k=>first(v) for (k,v) in light_model) #Get only the first light
    return light_model
end

function extract_light_data(data, index)
    #now just works for one light, takes index
    # Initialize arrays to hold extracted light data
    light_xdir = []
    light_xpos = []

    # Extract light data for the given index
    light_xdir_value = data.light_xdir[index, :]
    light_xpos_value = data.light_xpos[index, :]
    return Float32.(light_xdir_value), Float32.(light_xpos_value)
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


function extract_camera_model(model)
    # Iterate over each row in model.cam_pos
    for i in 1:size(model.cam_pos, 1)
        camera_position = model.cam_pos[i, :]
        camera_fovy = model.cam_fovy[i]
        #can add more if needed, but just fovy used in python code
        return camera_fovy
    end
end