const LIBEGL = "./libegl_example.so"
function setup_egl(width::Cint, height::Cint)
    # `setup_egl` now takes two integer arguments: width and height
    egl_status = ccall((:setup_egl, LIBEGL), Cint, (Cint, Cint), width, height)
    if egl_status != 0
        error("Failed to initialize EGL. Return code: $egl_status.")
    end
end

function compile_shader(source_file, shader_type)
    shader_source = open(read, source_file, "r") |> Vector{UInt8}
    shader = glCreateShader(shader_type)
	glShaderSource(shader, 1, Ptr{UInt8}[pointer(shader_source)], Ref{GLint}(length(shader_source)))  # nicer thanks to GLAbstraction
	glCompileShader(shader)
	# Check that it compiled correctly
	status = Ref(GLint(0))
	glGetShaderiv(shader, GL_COMPILE_STATUS, status)
	if status[] != GL_TRUE
		buffer = zeros(UInt8, 512)
		glGetShaderInfoLog(shader, 512, C_NULL, buffer)
		@error "$(unsafe_string(pointer(buffer), 512))"
	end
    return shader
end

function compile_shaders()
    vertex_shader = compile_shader("vertex.glsl", GL_VERTEX_SHADER)
    fragment_shader = compile_shader("fragment.glsl", GL_FRAGMENT_SHADER)

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

function to_cpu_array(width::Int32, height::Int32)
    glReadBuffer(GL_FRONT)

    pixels = Vector{UInt8}(undef, width * height * 3)

    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels)

	image_data = reshape(pixels, (3, width, height))
	image_data = reverse(image_data, dims=3)
end
