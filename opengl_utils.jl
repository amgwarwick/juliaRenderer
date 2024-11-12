# Load the shared library
const LIBEGL = "./libegl_example.so"

function setup_egl(width::Cint, height::Cint)
    # `setup_egl` now takes two integer arguments: width and height
    return ccall((:setup_egl, LIBEGL), Cint, (Cint, Cint), width, height)
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
