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


tex1 = extract_textures(model)[:, :, :, 3]
Images.colorview(Images.RGBA, tex1 ./ 255.0f0)