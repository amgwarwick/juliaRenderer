using StaticArrays
n_geoms::Int32 = model.ngeom
geom_types = model.geom_type
geom_sizes = model.geom_size
geom_rgbas = model.geom_rgba
geom_mater = model.geom_matid


# Loop over geometries
for i in 1:n_geoms
    geom_type = MuJoCo.mjtGeom(geom_types[i])
    rgba = SVector{4, Float32}(view(geom_rgbas, i, :))
    material_id = geom_mater[i]
    mat_tex_id = material_id != -1 ? model.mat_texid[material_id+1] : nothing
    mat_rgba = material_id != -1 ? model.mat_rgba[material_id+1] : nothing
    @info (; i, geom_type, rgba, material_id, mat_tex_id, mat_rgba)
end