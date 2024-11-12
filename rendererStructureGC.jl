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

# Load the shared library
const LIBEGL = "./libegl_example.so"

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
    n_envs::Int32

    egl_ctx_success::Int32

    projection_matrix::Matrix{Float32}

    light_model::Dict{String, Any}
    #textures when ready

    n_geoms_by_type::Dict{String, Int32}

    renderers::Vector{GeomRenderer}
end


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

function render(BatchRenderer, datas)

    instance_data_boxes, instance_data_spheres, instance_data_planes,  instance_data_capsules = extract_geom_data(model, datas, BatchRenderer.n_envs, BatchRenderer.n_geoms_by_type)

    camera_data_pos, camera_data_mat = extract_camera_data(datas[1]) #will change
    camera_data_pos = Float32.(camera_data_pos)

    light_data_xdir, light_data_xpos = extract_light_data(datas[1], 1) #will change

    glEnable(GL_DEPTH_TEST)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    display(BatchRenderer.renderers[1], instance_data_boxes, light_data_xdir, light_data_xpos, camera_data_mat, camera_data_pos, BatchRenderer)
    display(BatchRenderer.renderers[2], instance_data_spheres, light_data_xdir, light_data_xpos, camera_data_mat, camera_data_pos, BatchRenderer)
    display(BatchRenderer.renderers[3], instance_data_capsules, light_data_xdir, light_data_xpos, camera_data_mat, camera_data_pos, BatchRenderer)
    display(BatchRenderer.renderers[4], instance_data_planes, light_data_xdir, light_data_xpos, camera_data_mat, camera_data_pos, BatchRenderer)

    return save_egl_image("rendered_image_julia.png", BatchRenderer.n_envs*BatchRenderer.res, BatchRenderer.res)
    
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

function compile_shaders()
    vertex_source = open(read, "vertex.glsl", "r") |> Vector{UInt8}
    fragment_source = open(read, "fragment.glsl", "r") |> Vector{UInt8}

	# Compile the vertex shader
	vertex_shader = glCreateShader(GL_VERTEX_SHADER)
	glShaderSource(vertex_shader, 1, Ptr{UInt8}[pointer(vertex_source)], Ref{GLint}(length(vertex_source)))  # nicer thanks to GLAbstraction
	glCompileShader(vertex_shader)
	# Check that it compiled correctly
	status = Ref(GLint(0))
	glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, status)
	if status[] != GL_TRUE
		buffer = Array(UInt8, 512)
		glGetShaderInfoLog(vertex_shader, 512, C_NULL, buffer)
		@error "$(unsafe_string(pointer(buffer), 512))"
	end

	# Compile the fragment shader
	fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
	glShaderSource(fragment_shader, 1, Ptr{UInt8}[pointer(fragment_source)], Ref{GLint}(length(fragment_source)))  # nicer thanks to GLAbstraction
	glCompileShader(fragment_shader)
	# Check that it compiled correctly
	status = Ref(GLint(0))
	glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, status)
	if status[] != GL_TRUE
		buffer = Array(UInt8, 512)
		glGetShaderInfoLog(fragment_shader, 512, C_NULL, buffer)
		@error "$(unsafe_string(pointer(buffer), 512))"
	end

	# Connect the shaders by combining them into a program
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
