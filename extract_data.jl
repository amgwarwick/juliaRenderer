using StaticArrays

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

function extract_geom_data(model, datas, n_env, countgeoms)

	matrix_instance_data_boxes = zeros(Float32, 22, n_env * countgeoms["nboxes"])
	matrix_instance_data_spheres = zeros(Float32, 22, n_env * countgeoms["nspheres"])
	matrix_instance_data_capsules = zeros(Float32, 22, n_env * countgeoms["ncapsules"])
	matrix_instance_data_planes = zeros(Float32, 22, n_env * countgeoms["nplanes"])
	
    # Initialize counters
    for j in 1:n_env
        box_counter::Int32 = 0
        sphere_counter::Int32 = 0
        capsule_counter::Int32 = 0
        plane_counter::Int32 = 0

        #scene inside
        #scn = MuJoCo.VisualiserScene()
        #cam = MuJoCo.VisualiserCamera()
        #opt = MuJoCo.VisualiserOption()
        #pert = MuJoCo.VisualiserPerturb()
        #MuJoCo.mjv_makeScene(model, scn, 1000) #scn should now contain the scene object

        # mjCAT_ALL = 7
        #catmask = 7

        #mjv_updateScene(model, datas[j], opt, pert, cam, catmask, scn)

        #n_geoms = scn.ngeom

        n_geoms::Int32 = model.ngeom

        # Loop over geometries
        for i in 1:n_geoms
            # Compute the pointer to the current geom
            #geom_ptr = Ptr{MuJoCo.LibMuJoCo.mjvGeom_}(scn.geoms + i * sizeof(MuJoCo.LibMuJoCo.mjvGeom_))
            
            # Load the geom structure
            #geom = unsafe_load(geom_ptr)
            #geom_type = geom.type
            geom_type::Int32 = model.geom_type[i]
            #pos = Float32[geom.pos...]
			pos = SVector{3, Float32}(datas[j].geom_xpos[i, :])

            #xmat = collect(geom.mat)
            #xmat = Float32[xmat...]

            xmat = SVector{9, Float32}(datas[j].geom_xmat[i, :])

            rgba = SVector{4, Float32}(model.geom_rgba[i, :])

            offset::Int32 = j - 1

            #tex_id = Float32(geom.texid)
            #if tex_id != -1
                #tex_id = get(geomTexID_to_openglTexID, tex_id, -1)
                #tex_id = Float32(-1)
            #end

            tex_id::Int32 = -1

            # Initialize the geomsize array
            geomsize = SVector{3, Float32}(model.geom_size[i, :])

            if geom_type == 2
				geomsize = SVector{3, Float32}(geomsize[1], geomsize[1], geomsize[1])
            elseif geom_type == 0
                if geomsize[1] == 0
                    #geomsize[1] = 5.0
                    geomsize = SVector{3, Float32}(5.0, geomsize[2], geomsize[3])
                end
                if geomsize[2] == 0
                    #geomsize[2] = 5.0
                    geomsize = SVector{3, Float32}(geomsize[1], 5.0, geomsize[3])
                end
					geomsize = SVector{3, Float32}(geomsize[1], geomsize[2], 1.0)
            elseif geom_type == 3
                # for scene geomsize = Float32[geomsize[2], geomsize[2], geomsize[3]]
                geomsize = SVector{3, Float32}(geomsize[1], geomsize[1], geomsize[2]) #for model
            end

            # Compute the model matrix, M is pre-allocated and reused
            M = modelMatrix(pos, xmat, geomsize)

			if geom_type == 6
				box_idx::Int32 = (j - 1) * countgeoms["nboxes"] + box_counter + 1
				box_counter += 1
				# Reshape, transpose, and flatten the matrix
				matrix_instance_data_boxes[1:16, box_idx] = M
				matrix_instance_data_boxes[17:20, box_idx] = rgba
				matrix_instance_data_boxes[21, box_idx] = offset
				matrix_instance_data_boxes[22, box_idx] = tex_id

			elseif geom_type == 4 || geom_type == 2
				sphere_idx::Int32 = (j - 1) * countgeoms["nspheres"] + sphere_counter + 1
				sphere_counter += 1
				# Reshape, transpose, and flatten the matrix
				matrix_instance_data_spheres[1:16, sphere_idx] = M
				matrix_instance_data_spheres[17:20, sphere_idx] = rgba
				matrix_instance_data_spheres[21, sphere_idx] = offset
				matrix_instance_data_spheres[22, sphere_idx] = tex_id

			elseif geom_type == 0
				plane_idx::Int32 = (j - 1) * countgeoms["nplanes"] + plane_counter + 1
				plane_counter += 1
				# Reshape, transpose, and flatten the matrix
				matrix_instance_data_planes[1:16, plane_idx] = M
				matrix_instance_data_planes[17:20, plane_idx] = rgba
				matrix_instance_data_planes[21, plane_idx] = offset
				matrix_instance_data_planes[22, plane_idx] = tex_id

			elseif geom_type == 3
				capsule_idx::Int32 = (j - 1) * countgeoms["ncapsules"] + capsule_counter + 1
				capsule_counter += 1
				# Reshape, transpose, and flatten the matrix
				matrix_instance_data_capsules[1:16, capsule_idx] = M
				matrix_instance_data_capsules[17:20, capsule_idx] = rgba
				matrix_instance_data_capsules[21, capsule_idx] = offset
				matrix_instance_data_capsules[22, capsule_idx] = tex_id

                #move camera data and light data here

            end
        end
    end    

    #println(matrix_instance_data_boxes)
    #println(matrix_instance_data_spheres)
    #println(matrix_instance_data_planes)
    #println(matrix_instance_data_capsules)





    return matrix_instance_data_boxes, matrix_instance_data_spheres, matrix_instance_data_planes, matrix_instance_data_capsules
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
