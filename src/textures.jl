function extract_textures(model)
    widths = model.tex_width
    heights = model.tex_height
    n_textures = model.ntex
    tex_rgb_data = model.tex_rgb
    texture_stack = fill(UInt8(255), 4, 512, 512, n_textures)
    for i = 1:n_textures
        width = widths[i]
        height = heights[i]
        start_idx = model.tex_adr[i] + 1
        end_idx = start_idx + 3 * width * height - 1
        texture = reshape(tex_rgb_data[start_idx:end_idx], 3, width, height)
        permuted_texture = permutedims(texture, (2, 3, 1))
        resized_texture = Images.imresize(permuted_texture, (512, 512))
        scaled_texture = permutedims(resized_texture, (3, 1, 2))
        texture_stack[1:3, :, :, i] = round.(scaled_texture)
    end

    return texture_stack
end
#texture_stack = extract_textures(model)
function upload_textures(texture_stack::Array{UInt8, 4})
    texture_pointer_ref = Ref{GLuint}(0)
    glGenTextures(1, texture_pointer_ref)
    texture_pointer_ref[] > 0 || error("Failed to allocate texture stack")
    glBindTexture(GL_TEXTURE_2D_ARRAY, texture_pointer_ref[])
    _, height, width, n_textures = size(texture_stack)
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGB8, width, height, n_textures)
    
    rgb_only = texture_stack[1:3,:,:,:]
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, width, height, n_textures, GL_RGB, GL_UNSIGNED_BYTE, rgb_only)

    readback = zero(rgb_only) .+ eltype(rgb_only)(7) #fill(UInt8(255), 3, 512, 512)#
    glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RGB, GL_UNSIGNED_BYTE, readback)
    @assert readback == rgb_only

    buf = Ref{GLint}(-1)
    glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_DEPTH, buf)
  	@assert buf[] == n_textures

    #TODO: Set these to good values
    #glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
    #glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
    #glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE)
    #glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glGenerateMipmap(GL_TEXTURE_2D_ARRAY)

    return texture_pointer_ref[]
end

#tex1 = extract_textures(model)[:, :, :, 3]
#Images.colorview(Images.RGBA, tex1 ./ 255.0f0)